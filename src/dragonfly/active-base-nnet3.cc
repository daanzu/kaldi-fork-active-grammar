// NNet3 Active Base

// Copyright   2019  David Zurow

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License
// for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "feat/wave-reader.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice-lexicon.h"
#include "nnet3/nnet-utils.h"
#include "fst/script/compile.h"

#include "active-base-nnet3.h"
#include "active-replace-fst.h"

namespace dragonfly {

using namespace kaldi;
using namespace fst;

ActiveBaseNNet3OnlineModelWrapper::ActiveBaseNNet3OnlineModelWrapper(ActiveBaseNNet3OnlineModelConfig::Ptr config, int32 verbosity)
    : BaseNNet3OnlineModelWrapper(config, verbosity), config_(config) {
}

ActiveBaseNNet3OnlineModelWrapper::~ActiveBaseNNet3OnlineModelWrapper() {
}

bool ActiveBaseNNet3OnlineModelWrapper::SetMimicGrammarFst(int32 grammar_fst_index, StdFst* grammar_fst) {
    ExecutionTimer timer("SetMimicGrammarFst", 1);
    mimic_fsts_.erase(grammar_fst_index);
    static const std::vector<std::pair<StdArc::Label, StdArc::Label>> ilabels{ { word_syms_->Find(config_->eps_disambig_sym), 0 } };
    static const std::vector<std::pair<StdArc::Label, StdArc::Label>> olabels{ { word_syms_->Find("#nonterm:end"), 0 } };
    auto fst = StdRelabelFst(*grammar_fst, ilabels, olabels);
    mimic_fsts_.emplace(std::make_pair(grammar_fst_index, std::unique_ptr<StdFst>(new StdConstFst(std::forward<StdFst>(fst)))));
    // { StdVectorFst expanded_fst(*mimic_fsts_.at(grammar_fst_index)); expanded_fst.Write("tmp_mimic.fst"); }
    if (mimic_fsts_.size() > config_->max_num_rules) KALDI_ERR << "more grammars than max number";
    return true;
}

bool ActiveBaseNNet3OnlineModelWrapper::SetMimicDictationFst(StdFst* grammar_fst) {
    ExecutionTimer timer("SetMimicDictationFst", 1);
    // mimic_dictation_fst_.reset(CastOrConvertToConstFst(grammar_fst));
    // static const std::vector<std::pair<StdArc::Label, StdArc::Label>> ilabels{ { word_syms_->Find(config_->eps_disambig_sym), 0 } };
    // static const std::vector<std::pair<StdArc::Label, StdArc::Label>> olabels;  // Always empty, because only relabeling ilabels.
    // auto fst = StdRelabelFst(*grammar_fst, ilabels, olabels);
    // auto fst = StdProjectFst(*grammar_fst, PROJECT_OUTPUT);  // Faster than relabeling.
    // mimic_dictation_fst_.reset(new StdConstFst(fst));
    auto fst = new StdProjectFst(*grammar_fst, PROJECT_OUTPUT);  // Faster than relabeling.
    mimic_dictation_fst_.reset(fst);
    return true;
}

bool ActiveBaseNNet3OnlineModelWrapper::MimicInternal(const std::string& input, std::string* output_p, int32 grammar_fst_index) {
    ExecutionTimer timer("MimicInternal", 2);

    // Split input text up into labels.
    std::istringstream iss(input);
    std::vector<std::string> input_words(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());  // Split string by spaces.
    std::vector<int32> input_labels;
    for (const auto& word : input_words)
        input_labels.emplace_back(word_syms_->Find(word));
    timer.step("split input");

    std::vector<std::pair<int32, const StdFst*> > label_fst_pairs;
    auto rules_words_offset = word_syms_->Find("#nonterm:rule0");
    auto dictation_words_offset = word_syms_->Find("#nonterm:dictation");

    // Set up mimic FSTs.
    // if (mimic_fsts_.size() != grammar_fsts_.size())
    //     KALDI_WARN << "mismatched number of mimic_fsts_ and grammar_fsts_";
    for (const auto& it : mimic_fsts_)
        label_fst_pairs.emplace_back(rules_words_offset + it.first, it.second.get());
    if (mimic_dictation_fst_)
        label_fst_pairs.emplace_back(dictation_words_offset, mimic_dictation_fst_.get());
    timer.step("setup mimic");

    // Set up root FST.
    int64 root_fst_nonterm;
    StdVectorFst top_fst;  // May not be used/needed.
    if (grammar_fst_index == -1) {
        // Build top FST (to all FSTs) to be root FST.
        root_fst_nonterm = word_syms_->AvailableKey();

        auto start_state = top_fst.AddState();
        top_fst.SetStart(start_state);
        auto final_state = top_fst.AddState();
        top_fst.SetFinal(final_state, 0.0);
        for (const auto& it : mimic_fsts_)
            if (it.first < config_->max_num_exported_rules)  // Only include exported rules.
                top_fst.AddArc(start_state, StdArc(0, (rules_words_offset + it.first), 0.0, final_state));

        ArcSort(&top_fst, StdILabelCompare());
        label_fst_pairs.emplace_back(root_fst_nonterm, &top_fst);

    } else {
        // Use given grammar as root FST.
        root_fst_nonterm = rules_words_offset + grammar_fst_index;
    }
    timer.step("setup root");

    // Set up replace_fst.
    ActiveReplaceFstOptions<StdArc> replace_options(root_fst_nonterm, REPLACE_LABEL_OUTPUT, REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    // replace_options.take_ownership = true;  // true means FSTs are destructed upon ActiveReplaceFst destruction; default false means they are instead copied initially.
    auto replace_fst = ActiveReplaceFst<StdArc>(label_fst_pairs, replace_options);
    timer.step("setup replace_fst");

    std::set<int32> grammars_activity_by_label;  // Indexed by non-terminal label.
    for (auto rule_number : grammars_activity_)
        grammars_activity_by_label.insert(rule_number + rules_words_offset);
    if (mimic_dictation_fst_)
        grammars_activity_by_label.insert(dictation_words_offset);  // dictation_fst_ is only enabled if present
    replace_fst.UpdateActivity(grammars_activity_by_label);
    timer.step("setup activity");
    // { StdVectorFst expanded_fst(replace_fst); expanded_fst.Write("tmp_replace.fst"); }

    // Build linear automaton that accepts given input text.
    StdVectorFst input_fst;
    auto prev_state = input_fst.AddState();
    input_fst.SetStart(prev_state);
    for (auto label : input_labels) {
        auto state = input_fst.AddState();
        input_fst.AddArc(prev_state, StdArc(label, label, StdArc::Weight::One(), state));
        prev_state = state;
    }
    input_fst.SetFinal(prev_state, StdArc::Weight::One());
    timer.step("build input_fst");

    // Compose input recognizer with replace_fst that accepts the grammar, resulting in the accepted output (if any).
    auto composed_fst = StdComposeFst(input_fst, replace_fst);
    // static const std::vector<std::pair<StdArc::Label, StdArc::Label>> relabel_ilabels{ { word_syms_->Find(config_->eps_disambig_sym), 0 } };
    // static const std::vector<std::pair<StdArc::Label, StdArc::Label>> relabel_olabels;  // Always empty, because only relabeling ilabels.
    // auto composed_fst = StdComposeFst(input_fst, StdRelabelFst(replace_fst, relabel_ilabels, relabel_olabels));
    // { StdVectorFst expanded_fst(composed_fst); expanded_fst.Write("tmp_composed.fst"); }
    StdVectorFst output_fst;
    ShortestPath(composed_fst, &output_fst, 1);
    RmEpsilon(&output_fst);
    timer.step("build output_fst");
    if (output_fst.Start() == kNoStateId)
        return false;

    if (output_p != nullptr) {
        // Build output text from result of composition.
        TopSort(&output_fst);
        if (!output_fst.Properties(kTopSorted, false)) KALDI_ERR << "should be top sorted";
        std::vector<int32> output_labels;
        for (StateIterator<StdFst> siter(output_fst); !siter.Done(); siter.Next())
            for (ArcIterator<StdFst> aiter(output_fst, siter.Value()); !aiter.Done(); aiter.Next())
                output_labels.emplace_back(aiter.Value().olabel);
        *output_p = WordIdsToString(output_labels);
        timer.step("build output words");
    }
    return true;
}

} // namespace dragonfly


extern "C" {
#include "dragonfly.h"
}

using namespace dragonfly;

bool nnet3_active_base__set_mimic_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<ActiveBaseNNet3OnlineModelWrapper*>(model_vp);
    auto fst = static_cast<StdFst*>(grammar_fst_cp);
    return model->SetMimicGrammarFst(grammar_fst_index, fst);
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_active_base__set_mimic_dictation_fst_file(void* model_vp, const char* grammar_fst_filename_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<ActiveBaseNNet3OnlineModelWrapper*>(model_vp);
    auto fst = ReadFstKaldiGeneric(grammar_fst_filename_cp);
    return model->SetMimicDictationFst(fst);
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_active_base__mimic(void* model_vp, const char* input_cp, int32_t* grammars_activity_cp, uint32_t grammars_activity_cp_size,
    int32_t grammar_fst_index, char* output_cp, int32_t output_max_length) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<ActiveBaseNNet3OnlineModelWrapper*>(model_vp);
    if (grammars_activity_cp) {
        std::set<int32> grammars_activity(grammars_activity_cp, grammars_activity_cp + grammars_activity_cp_size);
        model->SetActiveGrammars(grammars_activity);
    }

    std::string input(input_cp);
    std::string output;
    auto output_p = (output_cp != nullptr) ? &output : nullptr;
    auto result = (grammar_fst_index == -1) ? model->Mimic(input, output_p) : model->MimicGrammar(input, output_p, grammar_fst_index);

    if (output_p) {
        strncpy(output_cp, output.c_str(), output_max_length);
        output_cp[output_max_length - 1] = 0;
        if (output.size() >= output_max_length)
            KALDI_WARN << "nnet3_active_base__mimic: output_max_length-1 < " << output.size();
    }
    return result;
    END_INTERFACE_CATCH_HANDLER(false)
}
