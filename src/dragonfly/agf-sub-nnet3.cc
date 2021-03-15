// NNet3 AGF

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
#include "decoder/active-grammar-fst.h"

#include "agf-sub-nnet3.h"
#include "compile-graph-agf.hh"
#include "active-cache.h"
#include "active-replace-fst.h"
#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;

AgfNNet3OnlineModelWrapper::AgfNNet3OnlineModelWrapper(AgfNNet3OnlineModelConfig::Ptr config, int32 verbosity)
    : ActiveBaseNNet3OnlineModelWrapper(config, verbosity), config_(config) {
    KALDI_VLOG(2) << "kNontermBigNumber, GetEncodingMultiple: " << kNontermBigNumber << ", " << GetEncodingMultiple(config_->nonterm_phones_offset);

    if ((config_->top_fst != 0) == !config_->top_fst_filename.empty()) KALDI_ERR << "AgfNNet3OnlineModelWrapper requires exactly one of top_fst and top_fst_filename";
    if (config_->top_fst != 0)
        top_fst_ = new StdConstFst(*static_cast<StdVectorFst*>((void*)config_->top_fst));
    if (!config_->top_fst_filename.empty())
        top_fst_ = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(config_->top_fst_filename));
    KALDI_VLOG(2) << "top_fst @ 0x" << top_fst_ << " " << top_fst_->NumStates() << " states";

    if (!config_->dictation_fst_filename.empty())
        dictation_fst_ = ReadFstFile(config_->dictation_fst_filename);

    auto first_rule_sym = word_syms_->Find("#nonterm:rule0"),
        last_rule_sym = first_rule_sym + 9999;
    rule_relabel_mapper_ = new CombineRuleNontermMapper<CompactLatticeArc>(first_rule_sym, last_rule_sym);

    if (enable_carpa_) KALDI_ERR << "AgfNNet3OnlineModelWrapper does not support carpa rescoring";
    if (enable_rnnlm_) KALDI_ERR << "AgfNNet3OnlineModelWrapper does not support rnnlm rescoring";
}

AgfNNet3OnlineModelWrapper::~AgfNNet3OnlineModelWrapper() {
    CleanupDecoder();
    delete top_fst_;
    delete dictation_fst_;
    delete active_grammar_fst_;
    delete rule_relabel_mapper_;
    // FIXME: grammar_fsts_ elements memory leak
    // FIXME: mimic_fsts_ elements memory leak
}

int32 AgfNNet3OnlineModelWrapper::AddGrammarFst(int32 grammar_fst_index, fst::StdConstFst* grammar_fst, std::string grammar_name) {
    InvalidateActiveGrammarFst();
    if (grammar_fst_index >= config_->max_num_rules) KALDI_ERR << "cannot add more than max number of rules";
    KALDI_VLOG(2) << "adding FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst->NumStates() << " states " << grammar_name;
    auto ok = grammar_fsts_.insert({grammar_fst_index, grammar_fst}).second;
    if (!ok) KALDI_ERR << "cannot add grammar to duplicate grammar_fst_index " << grammar_fst_index;
    grammar_fsts_name_map_[grammar_fst] = grammar_name;
    return grammar_fst_index;
}

int32 AgfNNet3OnlineModelWrapper::AddGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename) {
    auto grammar_fst = ReadFstFile(grammar_fst_filename);
    return AddGrammarFst(grammar_fst_index, grammar_fst, grammar_fst_filename);
}

bool AgfNNet3OnlineModelWrapper::ReloadGrammarFst(int32 grammar_fst_index, fst::StdConstFst* grammar_fst, std::string grammar_name) {
    InvalidateActiveGrammarFst();
    auto old_grammar_fst = grammar_fsts_.at(grammar_fst_index);
    grammar_fsts_name_map_.erase(old_grammar_fst);
    delete old_grammar_fst;

    KALDI_VLOG(2) << "reloading FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst->NumStates() << " states " << grammar_name;
    grammar_fsts_.at(grammar_fst_index) = grammar_fst;
    grammar_fsts_name_map_[grammar_fst] = grammar_name;
    return true;
}

bool AgfNNet3OnlineModelWrapper::ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename) {
    auto grammar_fst = ReadFstFile(grammar_fst_filename);
    return ReloadGrammarFst(grammar_fst_index, grammar_fst, grammar_fst_filename);
}

bool AgfNNet3OnlineModelWrapper::RemoveGrammarFst(int32 grammar_fst_index) {
    InvalidateActiveGrammarFst();
    auto grammar_fst = grammar_fsts_.at(grammar_fst_index);
    KALDI_VLOG(2) << "removing FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fsts_name_map_.at(grammar_fst);
    auto erased = grammar_fsts_.erase(grammar_fst_index);
    if (erased < 1) KALDI_ERR << "cannot find grammar_fst_index " << grammar_fst_index;
    grammar_fsts_name_map_.erase(grammar_fst);
    delete grammar_fst;
    return true;
}

bool AgfNNet3OnlineModelWrapper::InvalidateActiveGrammarFst() {
    if (DecoderReady(decoder_)) KALDI_ERR << "cannot modify/invalidate GrammarFst in the middle of decoding!";
    if (active_grammar_fst_) {
        delete active_grammar_fst_;
        active_grammar_fst_ = nullptr;
        return true;
    }
    return false;
}

void AgfNNet3OnlineModelWrapper::StartDecoding() {
    ExecutionTimer timer("StartDecoding", 2);
    ActiveBaseNNet3OnlineModelWrapper::StartDecoding();

    if (active_grammar_fst_ == nullptr) {
        std::vector<std::pair<int32, const StdConstFst *> > ifsts;
        for (auto grammar_fst_pair : grammar_fsts_) {
            auto grammar_fst_index = grammar_fst_pair.first;
            auto grammar_fst = grammar_fst_pair.second;
            int32 nonterm_phone = config_->rules_phones_offset + grammar_fst_index;
            ifsts.emplace_back(std::make_pair(nonterm_phone, grammar_fst));
        }
        if (dictation_fst_ != nullptr) {
            ifsts.emplace_back(std::make_pair(config_->dictation_phones_offset, dictation_fst_));
        }
        active_grammar_fst_ = new ActiveGrammarFst(config_->nonterm_phones_offset, *top_fst_, ifsts);
    }

    std::set<int32> grammars_activity;
    for (auto rule_number : grammars_activity_) grammars_activity.insert(rule_number + config_->rules_phones_offset);
    if (dictation_fst_ != nullptr) grammars_activity.insert(config_->dictation_phones_offset);  // dictation_fst_ is only enabled if present
    active_grammar_fst_->UpdateActivity(grammars_activity);

    decoder_ = new SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>(
        decoder_config_, trans_model_, *decodable_info_, *active_grammar_fst_, feature_pipeline_);
}

void AgfNNet3OnlineModelWrapper::CleanupDecoder() {
    delete decoder_;
    decoder_ = nullptr;
    ActiveBaseNNet3OnlineModelWrapper::CleanupDecoder();
}

bool AgfNNet3OnlineModelWrapper::Decode(BaseFloat samp_freq, const Vector<BaseFloat>& samples, bool finalize, bool save_adaptation_state) {
    if (!DecoderReady(decoder_))
        StartDecoding();
    return ActiveBaseNNet3OnlineModelWrapper::Decode(decoder_, samp_freq, samples, finalize, save_adaptation_state);
}

void AgfNNet3OnlineModelWrapper::GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) {
    ExecutionTimer timer("GetDecodedString", 2);

    decoded_string = "";
    if (likelihood) *likelihood = NAN;
    if (confidence) *confidence = NAN;
    if (expected_error_rate) *expected_error_rate = NAN;
    if (lm_score) *lm_score = NAN;
    if (am_score) *am_score = NAN;

    if (!decoder_) KALDI_ERR << "No decoder";
    if (decoder_->NumFramesDecoded() == 0) {
        if (decoder_finalized_) KALDI_WARN << "GetDecodedString on empty decoder";
        // else KALDI_VLOG(2) << "GetDecodedString on empty decoder";
        return;
    }

    Lattice best_path_lat;
    if (!decoder_finalized_) {
        // Decoding is not finished yet, so we will just look up the best partial result so far
        decoder_->GetBestPath(false, &best_path_lat);

    } else {
        decoder_->GetLattice(true, &decoded_clat_);
        if (decoded_clat_.NumStates() == 0) KALDI_ERR << "Empty decoded lattice";
        if (config_->lm_weight != 10.0)
            ScaleLattice(LatticeScale(config_->lm_weight, 10.0), &decoded_clat_);

        // WriteLattice(decoded_clat, "tmp/lattice");

        CompactLattice decoded_clat_relabeled = decoded_clat_;
        if (true) {
            // Relabel all nonterm:rules to nonterm:rule0, so redundant/ambiguous rules don't count as differing for measuring confidence
            ExecutionTimer timer("relabel");
            ArcMap(&decoded_clat_relabeled, rule_relabel_mapper_);
            // TODO: write a custom Visitor to coalesce the nonterm:rules arcs, and possibly erase them?
        }

        if (false || (true && (GetVerboseLevel() >= 1))) {
            // Difference between best path and second best path
            ExecutionTimer timer("confidence");
            int32 num_paths;
            // float conf = SentenceLevelConfidence(decoded_clat, &num_paths, NULL, NULL);
            std::vector<int32> best_sentence, second_best_sentence;
            float conf = SentenceLevelConfidence(decoded_clat_relabeled, &num_paths, &best_sentence, &second_best_sentence);
            timer.stop();
            KALDI_LOG << "SLC(" << num_paths << "paths): " << conf;
            if (num_paths >= 1) KALDI_LOG << "    1st best: " << WordIdsToString(best_sentence);
            if (num_paths >= 2) KALDI_LOG << "    2nd best: " << WordIdsToString(second_best_sentence);
            if (confidence) *confidence = conf;
        }

        if (false || (true && (GetVerboseLevel() >= 1))) {
            // Expected sentence error rate
            ExecutionTimer timer("expected_ser");
            MinimumBayesRiskOptions mbr_opts;
            mbr_opts.decode_mbr = false;
            MinimumBayesRisk mbr(decoded_clat_relabeled, mbr_opts);
            const vector<int32> &words = mbr.GetOneBest();
            // const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
            // const vector<pair<BaseFloat, BaseFloat> > &times = mbr.GetOneBestTimes();
            auto risk = mbr.GetBayesRisk();
            timer.stop();
            KALDI_LOG << "MBR(SER): " << risk << " : " << WordIdsToString(words);
            if (expected_error_rate) *expected_error_rate = risk;
        }

        if (false || (true && (GetVerboseLevel() >= 1))) {
            // Expected word error rate
            ExecutionTimer timer("expected_wer");
            MinimumBayesRiskOptions mbr_opts;
            mbr_opts.decode_mbr = true;
            MinimumBayesRisk mbr(decoded_clat_relabeled, mbr_opts);
            const vector<int32> &words = mbr.GetOneBest();
            // const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
            // const vector<pair<BaseFloat, BaseFloat> > &times = mbr.GetOneBestTimes();
            auto risk = mbr.GetBayesRisk();
            timer.stop();
            KALDI_LOG << "MBR(WER): " << risk << " : " << WordIdsToString(words);
            if (expected_error_rate) *expected_error_rate = risk;

            if (true) {
                ExecutionTimer timer("compare mbr");
                MinimumBayesRiskOptions mbr_opts;
                mbr_opts.decode_mbr = false;
                MinimumBayesRisk mbr_ser(decoded_clat_relabeled, mbr_opts);
                const vector<int32> &words_ser = mbr_ser.GetOneBest();
                timer.stop();
                if (mbr.GetBayesRisk() != mbr_ser.GetBayesRisk()) KALDI_WARN << "MBR risks differ";
                if (words != words_ser) KALDI_WARN << "MBR words differ";
            }
        }

        if (true) {
            // Use MAP (SER) as expected error rate
            ExecutionTimer timer("expected_error_rate");
            MinimumBayesRiskOptions mbr_opts;
            mbr_opts.decode_mbr = false;
            MinimumBayesRisk mbr(decoded_clat_relabeled, mbr_opts);
            // const vector<int32> &words = mbr.GetOneBest();
            if (expected_error_rate) *expected_error_rate = mbr.GetBayesRisk();
            // FIXME: also do confidence?
        }

        if (false) {
            CompactLattice pre_dictation_clat, in_dictation_clat, post_dictation_clat;
            auto nonterm_dictation = word_syms_->Find("#nonterm:dictation");
            auto nonterm_end = word_syms_->Find("#nonterm:end");
            bool ok;
            CopyDictationVisitor<CompactLatticeArc> visitor(&pre_dictation_clat, &in_dictation_clat, &post_dictation_clat, &ok, nonterm_dictation, nonterm_end);
            KALDI_ASSERT(ok);
            AnyArcFilter<CompactLatticeArc> filter;
            DfsVisit(decoded_clat_, &visitor, filter, true);
            WriteLattice(in_dictation_clat, "tmp/lattice_dict");
            WriteLattice(pre_dictation_clat, "tmp/lattice_dictpre");
            WriteLattice(post_dictation_clat, "tmp/lattice_dictpost");
        }

        CompactLatticeShortestPath(decoded_clat_, &best_path_clat_);
        ConvertLattice(best_path_clat_, &best_path_lat);
    } // if (decoder_finalized_)

    std::vector<int32> words;
    std::vector<int32> alignment;
    LatticeWeight weight;
    bool ok = GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
    if (!ok) KALDI_ERR << "GetLinearSymbolSequence returned false";

    int32 num_frames = alignment.size();
    // int32 num_words = words.size();
    if (lm_score) *lm_score = weight.Value1();
    if (am_score) *am_score = weight.Value2();
    if (likelihood) *likelihood = expf(-(*lm_score + *am_score) / num_frames);

    decoded_string = WordIdsToString(words);
}

bool AgfNNet3OnlineModelWrapper::SetMimicGrammarFst(int32 grammar_fst_index, StdConstFst* grammar_fst) {
    auto it = mimic_fsts_.find(grammar_fst_index);
    if (it != mimic_fsts_.end()) {
        // delete it->second;
    }
    mimic_fsts_.insert(it, {grammar_fst_index, grammar_fst});
    return true;
}

void TestReplace() {
    StdVectorFst top_fst;
    top_fst.AddState();
    top_fst.AddState();
    top_fst.AddState();
    top_fst.AddState();
    top_fst.AddState();
    top_fst.SetStart(0);
    top_fst.AddArc(0, StdArc(1, 1, StdArc::Weight::One(), 1));
    top_fst.AddArc(1, StdArc(100, 100, StdArc::Weight::One(), 2));
    top_fst.AddArc(2, StdArc(110, 110, StdArc::Weight::One(), 3));
    top_fst.AddArc(3, StdArc(2, 2, StdArc::Weight::One(), 4));
    top_fst.SetFinal(4, StdArc::Weight::One());

    StdVectorFst child_fst;
    child_fst.AddState();
    child_fst.AddState();
    child_fst.AddState();
    child_fst.AddState();
    child_fst.SetStart(0);
    child_fst.AddArc(0, StdArc(10, 10, StdArc::Weight::One(), 1));
    child_fst.AddArc(1, StdArc(11, 11, StdArc::Weight::One(), 2));
    child_fst.SetFinal(2, StdArc::Weight::One());

    std::vector<std::pair<int32, const StdFst*> > label_fst_pairs;
    label_fst_pairs.emplace_back(100, &child_fst);
    label_fst_pairs.emplace_back(99, &top_fst);

    fst::ReplaceFstOptions<StdArc> replace_options(99, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, 110);
    auto replace_fst = fst::ReplaceFst<StdArc>(label_fst_pairs, replace_options);

    StdConstFst expanded_fst(replace_fst);
}

bool AgfNNet3OnlineModelWrapper::Mimic(std::vector<int32>& ilabels, std::vector<int32>* olabels, int32 grammar_fst_index) {
    // TestReplace();
    std::vector<std::pair<int32, const StdFst*> > label_fst_pairs;
    auto rules_words_offset = word_syms_->Find("#nonterm:rule0");
    auto top_fst_nonterm = rules_words_offset + grammar_fst_index;

    if (mimic_fsts_.size() != grammar_fsts_.size())
        KALDI_WARN << "mismatched number of mimic_fsts_ and grammar_fsts_";
    for (auto it : mimic_fsts_)
        label_fst_pairs.emplace_back(rules_words_offset + it.first, it.second);
    if (dictation_fst_ != nullptr)
        label_fst_pairs.emplace_back(word_syms_->Find("#nonterm:dictation"), dictation_fst_);

    // fst::ReplaceFstOptions<StdArc> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    // auto replace_fst = fst::ReplaceFst<StdArc>(label_fst_pairs, replace_options);

    // fst::ReplaceFstOptions<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    // auto replace_fst = fst::ReplaceFst<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>>(label_fst_pairs, replace_options);
    // auto x = replace_fst.GetMutableImpl();
    // fst::ReplaceFstOptions<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    // auto replace_fst = fst::ActiveReplaceFst<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>>(label_fst_pairs, replace_options);
    // auto x = replace_fst;
    fst::ActiveReplaceFstOptions<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    auto replace_fst = fst::ActiveReplaceFst<StdArc, DefaultReplaceStateTable<StdArc>, DefaultActiveCacheStore<StdArc>>(label_fst_pairs, replace_options);

    StdVectorFst input_fst;
    auto prev_state = input_fst.AddState();
    input_fst.SetStart(prev_state);
    for (auto label : ilabels) {
        auto state = input_fst.AddState();
        input_fst.AddArc(prev_state, StdArc(label, label, StdArc::Weight::One(), state));
        prev_state = state;
    }
    input_fst.SetFinal(prev_state, StdArc::Weight::One());

    auto composed_fst = fst::ComposeFst<StdArc>(input_fst, replace_fst);
    if (composed_fst.Start() == kNoStateId)
        return false;

    // input_fst.Write("tmp_input.fst");
    // StdConstFst expanded_fst(replace_fst);
    // expanded_fst.Write("tmp_expanded.fst");
    // replace_fst.Write("tmp_replace.fst");
    // composed_fst.Write("tmp_composed.fst");

    if (olabels != nullptr) {
        StdVectorFst output_fst;
        fst::ShortestPath(composed_fst, &output_fst, 1);
        fst::RmEpsilon(&output_fst);
        fst::TopSort(&output_fst);
        if (!output_fst.Properties(fst::kTopSorted, false))
            KALDI_ERR << "should be top sorted";
        for (StateIterator<StdFst> siter(output_fst); !siter.Done(); siter.Next()) {
            for (ArcIterator<StdFst> aiter(output_fst, siter.Value()); !aiter.Done(); aiter.Next()) {
                olabels->emplace_back(aiter.Value().olabel);
            }
        }
    }
    return true;
}

} // namespace dragonfly


extern "C" {
#include "dragonfly.h"
}

using namespace dragonfly;

void* nnet3_agf__construct(char* model_dir_cp, char* config_str_cp, int32_t verbosity) {
    BEGIN_INTERFACE_CATCH_HANDLER
    std::string model_dir(model_dir_cp),
        config_str((config_str_cp != nullptr) ? config_str_cp : "");
    auto model = new AgfNNet3OnlineModelWrapper(AgfNNet3OnlineModelConfig::Create(model_dir, config_str), verbosity);
    return model;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

bool nnet3_agf__destruct(void* model_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    delete model;
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

int32_t nnet3_agf__add_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    auto fst = static_cast<StdVectorFst*>(grammar_fst_cp);
    auto const_fst = new StdConstFst(*fst);
    grammar_fst_index = model->AddGrammarFst(grammar_fst_index, const_fst);
    return grammar_fst_index;
    END_INTERFACE_CATCH_HANDLER(-1)
}

int32_t nnet3_agf__add_grammar_fst_file(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    grammar_fst_index = model->AddGrammarFst(grammar_fst_index, grammar_fst_filename);
    return grammar_fst_index;
    END_INTERFACE_CATCH_HANDLER(-1)
}

bool nnet3_agf__reload_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    auto fst = static_cast<StdVectorFst*>(grammar_fst_cp);
    auto const_fst = new StdConstFst(*fst);  // Newly-created FST, to be owned by the AgfNNet3OnlineModelWrapper, disentangled from the grammar_fst
    bool result = model->ReloadGrammarFst(grammar_fst_index, const_fst);
    return result;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_agf__reload_grammar_fst_file(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    bool result = model->ReloadGrammarFst(grammar_fst_index, grammar_fst_filename);
    return result;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_agf__remove_grammar_fst(void* model_vp, int32_t grammar_fst_index) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    bool result = model->RemoveGrammarFst(grammar_fst_index);
    return result;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_agf__decode(void* model_vp, float samp_freq, uint32_t num_samples, float* samples, bool finalize,
    int32_t* grammars_activity_cp, uint32_t grammars_activity_cp_size, bool save_adaptation_state) {
    BEGIN_INTERFACE_CATCH_HANDLER
    if (grammars_activity_cp_size) {
        auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::set<int32> grammars_activity(grammars_activity_cp, grammars_activity_cp + grammars_activity_cp_size);
        model->SetActiveGrammars(grammars_activity);
    }
    return nnet3_base__decode(model_vp, samp_freq, num_samples, samples, finalize, save_adaptation_state);
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_agf__set_mimic_grammar_fst(void* model_vp, int32_t grammar_fst_index, void* grammar_fst_cp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    auto fst = static_cast<StdVectorFst*>(grammar_fst_cp);
    auto const_fst = new StdConstFst(*fst);
    return model->SetMimicGrammarFst(grammar_fst_index, const_fst);
    END_INTERFACE_CATCH_HANDLER(false)
}

bool nnet3_agf__mimic(void* model_vp, int32_t target_labels_cp[], uint32_t target_labels_len, int32_t grammar_fst_index, int32_t output_labels_cp[], uint32_t* output_labels_len) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::vector<int32> ilabels(target_labels_cp, target_labels_cp + target_labels_len);
    std::vector<int32> olabels;
    auto olabels_p = (output_labels_cp != nullptr && output_labels_len != nullptr) ? &olabels : nullptr;
    auto result = model->Mimic(ilabels, olabels_p, grammar_fst_index);

    if (olabels_p) {
        for (auto i = 0; i < std::min((uint32_t)olabels.size(), *output_labels_len); ++i)
            output_labels_cp[i] = olabels[i];
        if (olabels.size() > *output_labels_len)
            KALDI_WARN << "nnet3_agf__mimic: output_labels_len < " << olabels.size();
        *output_labels_len = olabels.size();
    }
    return result;

    END_INTERFACE_CATCH_HANDLER(false)
}


void* nnet3_agf__construct_compiler(char* config_str_cp) {
    // FIXME: are we thread safe here?
    BEGIN_INTERFACE_CATCH_HANDLER
    std::string config_str((config_str_cp != nullptr) ? config_str_cp : "");
    auto config = nlohmann::json::parse(config_str).get<AgfCompilerConfig>();
    auto compiler = new AgfCompiler(config);
    return compiler;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

bool nnet3_agf__destruct_compiler(void* compiler_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto compiler = static_cast<AgfCompiler*>(compiler_vp);
    delete compiler;
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

void* nnet3_agf__compile_graph(void* compiler_vp, char* config_str_cp, void* grammar_fst_cp, bool return_graph) {
    // FIXME: are we thread safe here?
    BEGIN_INTERFACE_CATCH_HANDLER
    auto compiler = static_cast<AgfCompiler*>(compiler_vp);
    std::string config_str((config_str_cp != nullptr) ? config_str_cp : "");
    auto config = nlohmann::json::parse(config_str).get<AgfCompilerConfig>();
    auto fst = static_cast<StdFst*>(grammar_fst_cp);
    auto result = compiler->CompileGrammar(fst, &config);
    if (!return_graph) {
        if (config.hclg_wxfilename.empty())
            KALDI_WARN << "Compiled graph not saved to file or returned!";
        delete result;
        result = nullptr;
    }
    return result;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

void* nnet3_agf__compile_graph_text(void* compiler_vp, char* config_str_cp, char* grammar_fst_text_cp, bool return_graph) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto compiler = static_cast<AgfCompiler*>(compiler_vp);
    std::istringstream iss(grammar_fst_text_cp);
    auto fst = compiler->CompileFstText(iss);
    return nnet3_agf__compile_graph(compiler_vp, config_str_cp, fst, return_graph);
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

void* nnet3_agf__compile_graph_file(void* compiler_vp, char* config_str_cp, char* grammar_fst_filename_cp, bool return_graph) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto fst = ReadFstKaldiGeneric(grammar_fst_filename_cp);
    return nnet3_agf__compile_graph(compiler_vp, config_str_cp, fst, return_graph);
    END_INTERFACE_CATCH_HANDLER(nullptr)
}
