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
#include "fst/script/compile.h"

#include "laf-sub-nnet3.h"
#include "utils.h"
#include "kaldi-utils.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;

// static fst::FstRegisterer<StdOLabelLookAheadFst> StdOLabelLookAheadFst_registerer;

LafNNet3OnlineModelWrapper::LafNNet3OnlineModelWrapper(LafNNet3OnlineModelConfig::Ptr config, int32 verbosity)
    : BaseNNet3OnlineModelWrapper(config, verbosity), config_(config) {
    hcl_fst_ = fst::StdOLabelLookAheadFst::Read(config_->hcl_fst_filename);
    if (!ReadIntegerVectorSimple(config_->disambig_tids_filename, &disambig_tids_))
        KALDI_ERR << "cannot read disambig_tids file";

    if (config_->relabel_ilabels_filename.empty() == config_->word_syms_relabeled_filename.empty())
        KALDI_ERR << "must use exactly one of relabel_ilabels_filename and word_syms_relabeled_filename";
    if (!config_->relabel_ilabels_filename.empty()) {
        std::vector<std::vector<int32> > list;
        if (!ReadIntegerVectorVectorSimple(config_->relabel_ilabels_filename, &list))
            KALDI_ERR << "cannot read relabel_ilabels file";
        for (auto line : list) {
            if (line.size() != 2) KALDI_ERR << "badly formatted relabel_ilabels file";
            relabel_ilabels_.emplace_back(line[0], line[1]);
        }
    }
    if (!config_->word_syms_relabeled_filename.empty())
        if (!(word_syms_relabeled_ = fst::SymbolTable::ReadText(config_->word_syms_relabeled_filename)))
            KALDI_ERR << "cannot read word_syms_relabeled_filename";

    if (!config_->dictation_fst_filename.empty()) {
        if (!relabel_ilabels_.empty()) {
            KALDI_WARN << "relabeling dictation_fst is inefficient";
            ExecutionTimer timer("relabeling dictation_fst");
            auto fst = CastOrConvertToVectorFst(ReadFstKaldiGeneric(config_->dictation_fst_filename));
            // static const std::vector<std::pair<StdArc::Label, StdArc::Label>> olabels;  // Always empty
            // fst::Relabel(fst, relabel_ilabels_, olabels);
            // fst::ArcSort(fst, fst::StdILabelCompare());
            PrepareGrammarFst(fst);
            dictation_fst_ = CastOrConvertToConstFst(fst);
        } else
            dictation_fst_ = CastOrConvertToConstFst(ReadFstKaldiGeneric(config_->dictation_fst_filename));
    } else
        KALDI_WARN << "no dictation grammar";

    auto first_rule_sym = word_syms_->Find("#nonterm:rule0"),
        last_rule_sym = first_rule_sym + 9999;
    rule_relabel_mapper_ = new CombineRuleNontermMapper<CompactLatticeArc>(first_rule_sym, last_rule_sym);
}

LafNNet3OnlineModelWrapper::~LafNNet3OnlineModelWrapper() {
    CleanupDecoder();
    delete hcl_fst_;
    delete word_syms_relabeled_;
    delete dictation_fst_;
    delete rule_relabel_mapper_;
    // delete active_grammar_fst_;
}

// void LafNNet3OnlineModelWrapper::PendNonterm()

void LafNNet3OnlineModelWrapper::PrepareGrammarFst(fst::StdVectorFst* grammar_fst) {
    ExecutionTimer timer("PrepareGrammarFst");
    if (!relabel_ilabels_.empty()) {
        static const std::vector<std::pair<StdArc::Label, StdArc::Label>> olabels;  // Always empty
        fst::Relabel(grammar_fst, relabel_ilabels_, olabels);
        timer.step();
    }
    fst::ArcSort(grammar_fst, fst::StdILabelCompare());
    timer.step();

    // if (true) {
    //     {
    //         int32 nonterm = word_syms_->Find("#nonterm:rule0") + grammar_fst_index;
    //         VectorFst<StdArc> nonterm_fst;
    //         nonterm_fst.AddState();
    //         nonterm_fst.SetStart(0);
    //         nonterm_fst.AddState();
    //         nonterm_fst.SetFinal(1, 0.0);
    //         nonterm_fst.AddArc(0, StdArc(0, nonterm, 0.0, 1));
    //         fst::Concat(nonterm_fst, grammar_fst);
    //     }
    //     {
    //         int32 nonterm = word_syms_->Find("#nonterm:end");
    //         VectorFst<StdArc> nonterm_fst;
    //         nonterm_fst.AddState();
    //         nonterm_fst.SetStart(0);
    //         nonterm_fst.AddState();
    //         nonterm_fst.SetFinal(1, 0.0);
    //         nonterm_fst.AddArc(0, StdArc(0, nonterm, 0.0, 1));
    //         fst::Concat(grammar_fst, nonterm_fst);
    //     }
    //     timer.step();
    // }
}

int32 LafNNet3OnlineModelWrapper::AddGrammarFst(std::string& grammar_fst_filename) {
    auto grammar_fst = CastOrConvertToVectorFst(ReadFstKaldiGeneric(grammar_fst_filename));
    PrepareGrammarFst(grammar_fst);
    return AddGrammarFst(CastOrConvertToConstFst(grammar_fst), grammar_fst_filename);
}

int32 LafNNet3OnlineModelWrapper::AddGrammarFst(std::istream& grammar_text) {
    ExecutionTimer timer("AddGrammarFst:compiling");
    auto word_syms_maybe_relabeled = (word_syms_relabeled_) ? word_syms_relabeled_ : word_syms_;
    // FIXME: fix build for linux and macos, and CI for windows
    auto grammar_fstclass = fst::script::CompileFstInternal(grammar_text, "<AddGrammarFst>", "vector", "standard",
        word_syms_maybe_relabeled, word_syms_, nullptr, false, false, false, false, false);
    timer.step();
    auto grammar_fst = dynamic_cast<StdVectorFst*>(fst::Convert(*grammar_fstclass->GetFst<StdArc>(), "vector"));
    if (!grammar_fst) KALDI_ERR << "could not convert grammar Fst to StdVectorFst";
    timer.step();
    PrepareGrammarFst(grammar_fst);
    return AddGrammarFst(new fst::StdConstFst(*grammar_fst));
}

int32 LafNNet3OnlineModelWrapper::AddGrammarFst(fst::StdFst* grammar_fst, std::string grammar_name) {
    auto grammar_fst_index = grammar_fsts_.size();
    if (grammar_fst_index >= config_->max_num_rules) KALDI_ERR << "cannot add more than max number of rules";
    KALDI_VLOG(2) << "adding FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_name;
    grammar_fsts_.emplace_back(grammar_fst);
    grammar_fsts_name_map_[grammar_fst] = grammar_name;
    // if (active_grammar_fst_) {
    //     delete active_grammar_fst_;
    //     active_grammar_fst_ = nullptr;
    // }
    return grammar_fst_index;
}

bool LafNNet3OnlineModelWrapper::ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename) {
    auto old_grammar_fst = grammar_fsts_.at(grammar_fst_index);
    grammar_fsts_name_map_.erase(old_grammar_fst);
    delete old_grammar_fst;

    auto grammar_fst = ReadFstFile(grammar_fst_filename);
    KALDI_VLOG(2) << "reloading FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst_filename;
    grammar_fsts_.at(grammar_fst_index) = grammar_fst;
    grammar_fsts_name_map_[grammar_fst] = grammar_fst_filename;
    // if (active_grammar_fst_) {
    //     delete active_grammar_fst_;
    //     active_grammar_fst_ = nullptr;
    // }
    return true;
}

bool LafNNet3OnlineModelWrapper::RemoveGrammarFst(int32 grammar_fst_index) {
    auto grammar_fst = grammar_fsts_.at(grammar_fst_index);
    KALDI_VLOG(2) << "removing FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fsts_name_map_.at(grammar_fst);
    grammar_fsts_.erase(grammar_fsts_.begin() + grammar_fst_index);
    grammar_fsts_name_map_.erase(grammar_fst);
    delete grammar_fst;
    // if (active_grammar_fst_) {
    //     delete active_grammar_fst_;
    //     active_grammar_fst_ = nullptr;
    // }
    return true;
}

void LafNNet3OnlineModelWrapper::StartDecoding() {
    ExecutionTimer timer("StartDecoding", 2);
    BaseNNet3OnlineModelWrapper::StartDecoding();

    // if (active_grammar_fst_ == nullptr) {
    //     std::vector<std::pair<int32, const StdConstFst *> > ifsts;
    //     for (auto grammar_fst : grammar_fsts_) {
    //         int32 nonterm_phone = config_->rules_phones_offset + ifsts.size();
    //         ifsts.emplace_back(std::make_pair(nonterm_phone, grammar_fst));
    //     }
    //     if (dictation_fst_ != nullptr) {
    //         ifsts.emplace_back(std::make_pair(config_->dictation_phones_offset, dictation_fst_));
    //     }
    //     active_grammar_fst_ = new ActiveGrammarFst(config_->nonterm_phones_offset, *top_fst_, ifsts);
    // }

    // auto union_fst = fst::UnionFst<StdArc>(*grammar_fsts_[0], *grammar_fsts_[1]);
    // auto decode_fst = fst::LookaheadComposeFst(*hcl_fst_, union_fst, disambig_tids_);
    // auto decode_fst = fst::LookaheadComposeFst(*hcl_fst_, *dictation_fst_, disambig_tids_);

    std::vector<std::pair<int32, const StdFst *> > label_fst_pairs;
    auto rules_words_offset = word_syms_->Find("#nonterm:rule0");
    auto top_fst_nonterm = rules_words_offset + config_->max_num_rules + 1;

    // Build top_fst
    VectorFst<StdArc> top_fst;
    auto start_state = top_fst.AddState();
    top_fst.SetStart(start_state);
    auto final_state = top_fst.AddState();
    top_fst.SetFinal(final_state, 0.0);
    top_fst.SetFinal(start_state, 0.0);  // Allow start state to be final for no rule
    top_fst.AddArc(0, StdArc(0, 0, 0.0, final_state));  // Allow epsilon transition for no rule
    for (auto word : std::vector<std::string>{ "!SIL", "<unk>" })  // FIXME: make these configurable
        top_fst.AddArc(0, StdArc(word_syms_->Find(word), 0, 0.0, final_state));
    if (grammar_fsts_.size() > config_->max_num_rules) KALDI_ERR << "more grammars than max number";
    // for (size_t i = 0; i <= config_->max_num_rules; ++i) {
    for (size_t i = 0; i < grammar_fsts_.size(); ++i) {
        top_fst.AddArc(0, StdArc(0, (rules_words_offset + i), 0.0, final_state));
        // top_fst.AddArc(0, StdArc((rules_words_offset + i), (rules_words_offset + i), 0.0, final_state));
        label_fst_pairs.emplace_back((rules_words_offset + i), grammar_fsts_.at(i));
    }
    if (dictation_fst_ != nullptr)
        label_fst_pairs.emplace_back(word_syms_->Find("#nonterm:dictation"), dictation_fst_);
    // top_fst.AddArc(0, StdArc(0, word_syms_->Find("#nonterm:dictation"), 0.0, final_state));
    fst::ArcSort(&top_fst, fst::StdILabelCompare());

    label_fst_pairs.emplace_back(top_fst_nonterm, new fst::StdConstFst(top_fst));
    // for (size_t i = 0; i <= config_->max_num_rules; ++i)
    //     label_fst_pairs.emplace_back((rules_words_offset + i), grammar_fsts_.at(i));
    // for (auto grammar_fst : grammar_fsts_)
    //     label_fst_pairs.emplace_back((rules_words_offset + label_fst_pairs.size() - 1), grammar_fst);
    fst::ReplaceFstOptions<StdArc> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, word_syms_->Find("#nonterm:end"));
    auto replace_fst = fst::ReplaceFst<StdArc>(label_fst_pairs, replace_options);
    auto decode_fst = fst::LookaheadComposeFst(*hcl_fst_, replace_fst, disambig_tids_);

    auto grammars_activity = grammars_activity_;
    grammars_activity.push_back(dictation_fst_ != nullptr);  // dictation_fst_ is only enabled if present
    // active_grammar_fst_->UpdateActivity(grammars_activity);

    decoder_ = new SingleUtteranceNnet3DecoderTpl<fst::StdFst>(
        decoder_config_, trans_model_, *decodable_info_, *decode_fst, feature_pipeline_);
}

void LafNNet3OnlineModelWrapper::CleanupDecoder() {
    delete decoder_;
    decoder_ = nullptr;
    BaseNNet3OnlineModelWrapper::CleanupDecoder();
}

bool LafNNet3OnlineModelWrapper::Decode(BaseFloat samp_freq, const Vector<BaseFloat>& samples, bool finalize, bool save_adaptation_state) {
    if (!DecoderReady(decoder_))
        StartDecoding();
    return BaseNNet3OnlineModelWrapper::Decode(decoder_, samp_freq, samples, finalize, save_adaptation_state);
}

// grammars_activity is ignored once decoding has already started
bool LafNNet3OnlineModelWrapper::Decode(BaseFloat samp_freq, const Vector<BaseFloat>& samples, bool finalize,
        const std::vector<bool>& grammars_activity, bool save_adaptation_state) {
    SetActiveGrammars(std::move(grammars_activity));
    return Decode(samp_freq, samples, finalize, save_adaptation_state);
}

void LafNNet3OnlineModelWrapper::GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) {
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

} // namespace dragonfly


extern "C" {
#include "dragonfly.h"
}

using namespace dragonfly;

void* init_laf_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity) {
    std::string model_dir(model_dir_cp),
        config_str((config_str_cp != nullptr) ? config_str_cp : "");
    auto model = new LafNNet3OnlineModelWrapper(LafNNet3OnlineModelConfig::Create(model_dir, config_str), verbosity);
    return model;
}

int32_t add_grammar_fst_laf_nnet3(void* model_vp, char* grammar_fst_cp) {
    auto model = static_cast<LafNNet3OnlineModelWrapper*>(model_vp);
    // int32_t grammar_fst_index = model->AddGrammarFst(std::string(grammar_fst_cp));
    std::istringstream iss(grammar_fst_cp);
    int32_t grammar_fst_index = model->AddGrammarFst(iss);
    return grammar_fst_index;
}

bool reload_grammar_fst_laf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp) {
    auto model = static_cast<LafNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    bool result = model->ReloadGrammarFst(grammar_fst_index, grammar_fst_filename);
    return result;
}

bool remove_grammar_fst_laf_nnet3(void* model_vp, int32_t grammar_fst_index) {
    auto model = static_cast<LafNNet3OnlineModelWrapper*>(model_vp);
    bool result = model->RemoveGrammarFst(grammar_fst_index);
    return result;
}

bool decode_laf_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state) {
    if (grammars_activity_cp_size) {
        auto model = static_cast<LafNNet3OnlineModelWrapper*>(model_vp);
        std::vector<bool> grammars_activity(grammars_activity_cp_size, false);
        for (size_t i = 0; i < grammars_activity_cp_size; i++)
            grammars_activity[i] = grammars_activity_cp[i];
        model->SetActiveGrammars(std::move(grammars_activity));
    }
    return decode_base_nnet3(model_vp, samp_freq, num_samples, samples, finalize, save_adaptation_state);
}
