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

extern "C" {
#include "dragonfly.h"
}

#include <iomanip>

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

#include "nlohmann_json.hpp"
#include "utils.h"

#define DEFAULT_VERBOSITY 0

namespace dragonfly {

using namespace kaldi;
using namespace fst;

void WriteLattice(const CompactLattice clat_in, std::string name = "lattice") {
    auto clat = clat_in;
    RemoveAlignmentsFromCompactLattice(&clat);
    Lattice lat;
    ConvertLattice(clat, &lat);  // Convert to non-compact form.. won't introduce extra states because already removed alignments.
    StdVectorFst fst;
    ConvertLattice(lat, &fst);  // This adds up the (lm,acoustic) costs to get the normal (tropical) costs.
    Project(&fst, fst::PROJECT_OUTPUT);  // Because in the standard Lattice format, the words are on the output, and we want the word labels.
    RemoveEpsLocal(&fst);

    auto time = std::time(nullptr);
    std::stringstream ss;
    auto filename = name + "_%Y-%m-%d_%H-%M-%S.fst";
    ss << std::put_time(std::localtime(&time), filename.c_str());
    WriteFstKaldi(fst, ss.str());
    KALDI_WARN << "Wrote " << filename;
}

// ConstFst<StdArc>* CastOrConvertToConstFst(Fst<StdArc>* fst) {
//     // This version currently supports ConstFst<StdArc> or VectorFst<StdArc>
//     std::string real_type = fst->Type();
//     KALDI_ASSERT(real_type == "vector" || real_type == "const");
//     if (real_type == "const") {
//         return dynamic_cast<ConstFst<StdArc>*>(fst);
//     } else {
//         // As the 'fst' can't cast to ConstFst, we carete a new
//         // ConstFst<StdArc> initialized by 'fst', and delete 'fst'.
//         ConstFst<StdArc>* new_fst = new ConstFst<StdArc>(*fst);
//         delete fst;
//         return new_fst;
//     }
// }

template <class S>
class DictationOrderQueue : public QueueBase<S> {
   public:
    using StateId = S;

    DictationOrderQueue()
        : QueueBase<StateId>(OTHER_QUEUE), front_(0), back_(kNoStateId) {}

    virtual ~DictationOrderQueue() = default;

    StateId Head() const final { return front_; }

    void Enqueue(StateId s) final {
        if (front_ > back_) {
            front_ = back_ = s;
        } else if (s > back_) {
            back_ = s;
        } else if (s < front_) {
            front_ = s;
        }
        while (enqueued_.size() <= s) enqueued_.push_back(false);
        enqueued_[s] = true;
    }

    void Dequeue() final {
        enqueued_[front_] = false;
        while ((front_ <= back_) && (enqueued_[front_] == false)) ++front_;
    }

    void Update(StateId) final {}

    bool Empty() const final { return front_ > back_; }

    void Clear() final {
        for (StateId i = front_; i <= back_; ++i) enqueued_[i] = false;
        front_ = 0;
        back_ = kNoStateId;
    }

   private:
    StateId front_;
    StateId back_;
    std::vector<bool> enqueued_;
};

// template <class Arc>
// class CopyDictationVisitor {
//    public:
//     using StateId = typename Arc::StateId;

//     CopyDictationVisitor(T* return_data) {}

//     // Invoked before DFS visit.
//     void InitVisit(const Fst<Arc>& fst);

//     // Invoked when state discovered (2nd arg is DFS tree root).
//     bool InitState(StateId s, StateId root);

//     // Invoked when tree arc to white/undiscovered state examined.
//     bool TreeArc(StateId s, const Arc& arc);

//     // Invoked when back arc to grey/unfinished state examined.
//     bool BackArc(StateId s, const Arc& arc) {  }

//     // Invoked when forward or cross arc to black/finished state examined.
//     bool ForwardOrCrossArc(StateId s, const Arc& arc);

//     // Invoked when state finished ('s' is tree root, 'parent' is kNoStateId,
//     // and 'arc' is nullptr).
//     void FinishState(StateId s, StateId parent, const Arc* arc);

//     // Invoked after DFS visit.
//     void FinishVisit();
// };

template <class Arc>
class CopyDictationVisitor {
   public:
    using StateId = typename Arc::StateId;
    using Label = typename Arc::Label;

    CopyDictationVisitor(MutableFst<Arc> *ofst_pre_dictation, MutableFst<Arc> *ofst_in_dictation, MutableFst<Arc> *ofst_post_dictation,
            bool* ok, Label dictation_label, Label end_label)
        : ofst_pre_dictation_(ofst_pre_dictation), ofst_in_dictation_(ofst_in_dictation), ofst_post_dictation_(ofst_post_dictation),
        ok_(ok), dictation_label_(dictation_label), end_label_(end_label) {}

    void InitVisit(const ExpandedFst<Arc>& ifst) {
        ifst_ = &ifst;
        auto num_states = ifst_->NumStates();
        std::vector<MutableFst<Arc>*> ofsts = {ofst_pre_dictation_, ofst_in_dictation_, ofst_post_dictation_};
        for (auto ofst : ofsts) {
            ofst->DeleteStates();
            KALDI_ASSERT(ofst->AddState() == 0);
            ofst->SetStart(0);  // Dictation can't start at initial state
            while (ofst->NumStates() <= num_states) ofst->AddState();
        }
        state_map_.resize(num_states, StateType::Unknown);
        *ok_ = true;
    }

    bool InitState(StateId state, StateId root) {
        if (state == root) {
            state_map_[state] = StateType::PreDictation;
        }
        return true;
    }

    bool TreeArc(StateId state, const Arc& arc) {
        // Arc callback is called before State callback
        if ((state_map_[state] == StateType::PreDictation) && (arc.ilabel == dictation_label_)) {
            // Transition Pre->In
            // KALDI_ASSERT(state_map_[state] == StateType::PreDictation);
            ofst_pre_dictation_->AddArc(state, arc);
            ofst_pre_dictation_->SetFinal(arc.nextstate, Arc::Weight::One());
            ofst_in_dictation_->AddArc(0, Arc(0, 0, Arc::Weight::One(), arc.nextstate));
            state_map_[arc.nextstate] = StateType::InDictation;
        } else if ((state_map_[state] == StateType::InDictation) && (arc.ilabel == end_label_)) {
            // Transition In->Post
            // KALDI_ASSERT(state_map_[state] == StateType::InDictation);
            ofst_in_dictation_->SetFinal(state, Arc::Weight::One());
            ofst_post_dictation_->AddArc(0, Arc(0, 0, Arc::Weight::One(), state));
            ofst_post_dictation_->AddArc(state, arc);
            state_map_[arc.nextstate] = StateType::PostDictation;
        } else switch (state_map_[state]) {
            // Copy arc within same segment
            case StateType::PreDictation:
                ofst_pre_dictation_->AddArc(state, arc);
                state_map_[arc.nextstate] = StateType::PreDictation;
                break;
            case StateType::InDictation:
                ofst_in_dictation_->AddArc(state, arc);
                state_map_[arc.nextstate] = StateType::InDictation;
                break;
            case StateType::PostDictation:
                ofst_post_dictation_->AddArc(state, arc);
                state_map_[arc.nextstate] = StateType::PostDictation;
                break;
            case StateType::Unknown:
                KALDI_ASSERT(false);
        }
        return true;
    }

    bool ForwardOrCrossArc(StateId state, const Arc& arc) { return TreeArc(state, arc); }

    bool BackArc(StateId, const Arc&) { return (*ok_ = false); }  // None in a lattice!

    void FinishState(StateId state, StateId parent, const Arc* arc) {
        KALDI_ASSERT(state_map_[state] != StateType::Unknown);
        if (state_map_[state] == StateType::PostDictation)
            ofst_post_dictation_->SetFinal(state, ifst_->Final(state));
    }

    void FinishVisit() {}

   private:
    const ExpandedFst<Arc>* ifst_;
    MutableFst<Arc>* ofst_pre_dictation_;
    MutableFst<Arc>* ofst_in_dictation_;
    MutableFst<Arc>* ofst_post_dictation_;
    bool* ok_;
    Label dictation_label_;
    Label end_label_;
    // std::vector<bool> in_dictation_;
    // std::vector<bool> after_dictation_;

	enum class StateType : uint8 { Unknown, PreDictation, InDictation, PostDictation };
    std::vector<StateType> state_map_;
    // static constexpr uint8 kStateUnknown = 0;
    // static constexpr uint8 kStatePreDictation = 1;
    // static constexpr uint8 kStateInDictation = 2;
    // static constexpr uint8 kStatePostDictation = 3;
};

// template <class A>
// class CopyDictationVisitor1 : public CopyVisitor<A> {
//     public:
//         using Arc = A;
//         using StateId = typename Arc::StateId;

//         explicit CopyDictationVisitor1(MutableFst<Arc>* ofst) : ifst_(nullptr), ofst_(ofst) {}

//         void InitVisit(const Fst<A>& ifst) {
//             CopyVisitor<A>::InitVisit(ifst);
//             // std::stack<> stack;
//         }

//         bool InitState(StateId state, StateId) {
//             while (ofst_->NumStates() <= state) ofst_->AddState();
//             return true;
//         }

//         bool WhiteArc(StateId state, const Arc& arc) {
//             if () return CopyVisitor<A>::WhiteArc(state, arc);
//             return true;
//         }

//         bool GreyArc(StateId state, const Arc& arc) {
//             ofst_->AddArc(state, arc);
//             return true;
//         }

//         bool BlackArc(StateId state, const Arc& arc) {
//             ofst_->AddArc(state, arc);
//             return true;
//         }

//         void FinishState(StateId state) {
//             // ofst_->SetFinal(state, ifst_->Final(state));
//         }

//         void FinishVisit() {}

//     private:
//         bool copying_ = false;
// };


struct AgfNNet3OnlineModelConfig {

    BaseFloat beam = 14.0;  // normally 7.0
    int32 max_active = 14000;  // normally 7000
    int32 min_active = 200;
    BaseFloat lattice_beam = 8.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_weight = 7.0;  // 10.0 would be "neutral", with no scaling
    BaseFloat silence_weight = 1.0;  // default means silence weighting disabled
    int32 frame_subsampling_factor = 3;
    std::string model_dir;
    std::string mfcc_config_filename;
    std::string ie_config_filename;
    std::string model_filename;
    int32 nonterm_phones_offset = -1;  // offset from start of phones that start of nonterms are
    int32 rules_phones_offset = -1;  // offset from start of phones that the dictation nonterms are
    int32 dictation_phones_offset = -1;  // offset from start of phones that the kaldi_rules nonterms are
    std::string silence_phones_str = "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15";  // FIXME: from lang/phones/silence.csl
    std::string word_syms_filename;
    std::string word_align_lexicon_filename;
    std::string top_fst_filename;
    std::string dictation_fst_filename;

    bool Set(const std::string& name, const nlohmann::json& value) {
        if (name == "beam") { beam = value.get<BaseFloat>(); return true; }
        if (name == "max_active") { max_active = value.get<int32>(); return true; }
        if (name == "min_active") { min_active = value.get<int32>(); return true; }
        if (name == "lattice_beam") { lattice_beam = value.get<BaseFloat>(); return true; }
        if (name == "acoustic_scale") { acoustic_scale = value.get<BaseFloat>(); return true; }
        if (name == "lm_weight") { lm_weight = value.get<BaseFloat>(); return true; }
        if (name == "silence_weight") { silence_weight = value.get<BaseFloat>(); return true; }
        if (name == "frame_subsampling_factor") { frame_subsampling_factor = value.get<int32>(); return true; }
        if (name == "model_dir") { model_dir = value.get<std::string>(); return true; }
        if (name == "mfcc_config_filename") { mfcc_config_filename = value.get<std::string>(); return true; }
        if (name == "ie_config_filename") { ie_config_filename = value.get<std::string>(); return true; }
        if (name == "model_filename") { model_filename = value.get<std::string>(); return true; }
        if (name == "nonterm_phones_offset") { nonterm_phones_offset = value.get<int32>(); return true; }
        if (name == "rules_phones_offset") { rules_phones_offset = value.get<int32>(); return true; }
        if (name == "dictation_phones_offset") { dictation_phones_offset = value.get<int32>(); return true; }
        if (name == "silence_phones_str") { silence_phones_str = value.get<std::string>(); return true; }
        if (name == "word_syms_filename") { word_syms_filename = value.get<std::string>(); return true; }
        if (name == "word_align_lexicon_filename") { word_align_lexicon_filename = value.get<std::string>(); return true; }
        if (name == "top_fst_filename") { top_fst_filename = value.get<std::string>(); return true; }
        if (name == "dictation_fst_filename") { dictation_fst_filename = value.get<std::string>(); return true; }
        return false;
    }

    std::string ToString() {
        stringstream ss;
        ss << "AgfNNet3OnlineModelConfig...";
        ss << "\n    " << "beam: " << beam;
        ss << "\n    " << "max_active: " << max_active;
        ss << "\n    " << "min_active: " << min_active;
        ss << "\n    " << "lattice_beam: " << lattice_beam;
        ss << "\n    " << "acoustic_scale: " << acoustic_scale;
        ss << "\n    " << "lm_weight: " << lm_weight;
        ss << "\n    " << "silence_weight: " << silence_weight;
        ss << "\n    " << "frame_subsampling_factor: " << frame_subsampling_factor;
        ss << "\n    " << "model_dir: " << model_dir;
        ss << "\n    " << "mfcc_config_filename: " << mfcc_config_filename;
        ss << "\n    " << "ie_config_filename: " << ie_config_filename;
        ss << "\n    " << "model_filename: " << model_filename;
        ss << "\n    " << "nonterm_phones_offset: " << nonterm_phones_offset;
        ss << "\n    " << "rules_phones_offset: " << rules_phones_offset;
        ss << "\n    " << "dictation_phones_offset: " << dictation_phones_offset;
        ss << "\n    " << "silence_phones_str: " << silence_phones_str;
        ss << "\n    " << "word_syms_filename: " << word_syms_filename;
        ss << "\n    " << "word_align_lexicon_filename: " << word_align_lexicon_filename;
        ss << "\n    " << "top_fst_filename: " << top_fst_filename;
        ss << "\n    " << "dictation_fst_filename: " << dictation_fst_filename;
        return ss.str();
    }
};

class AgfNNet3OnlineModelWrapper {
    public:

        AgfNNet3OnlineModelWrapper(const std::string& model_dir, const std::string& config_str = "", int32 verbosity = DEFAULT_VERBOSITY);
        ~AgfNNet3OnlineModelWrapper();

        bool LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename);
        int32 AddGrammarFst(std::string& grammar_fst_filename);
        bool ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename);
        bool RemoveGrammarFst(int32 grammar_fst_index);
        bool SaveAdaptationState();
        void ResetAdaptationState();
        bool Decode(BaseFloat samp_freq, const Vector<BaseFloat>& frames, bool finalize, std::vector<bool>& grammars_activity, bool save_adaptation_state = true);

        void GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate);
        bool GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps);

    protected:

        AgfNNet3OnlineModelConfig config_;

        // Model
        fst::SymbolTable *word_syms = nullptr;
        std::vector<std::vector<int32> > word_align_lexicon;  // for each word, its word-id + word-id + a list of its phones
        StdConstFst *top_fst = nullptr;
        StdConstFst *dictation_fst = nullptr;
        std::vector<StdConstFst*> grammar_fsts;
        std::map<StdConstFst*, std::string> grammar_fsts_filename_map;  // maps grammar_fst -> name; for debugging
        // INVARIANT: same size: grammar_fsts, grammar_fsts_filename_map, grammar_fsts_enabled

        // Model objects
        OnlineNnet2FeaturePipelineConfig feature_config;
        nnet3::NnetSimpleLoopedComputationOptions decodable_config;
        LatticeFasterDecoderConfig decoder_config;
        OnlineEndpointConfig endpoint_config;
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        OnlineNnet2FeaturePipelineInfo* feature_info = nullptr;  // TODO: doesn't really need to be dynamically allocated (pointer)
        nnet3::DecodableNnetSimpleLoopedInfo* decodable_info = nullptr;  // contains precomputed stuff that is used by all decodable objects
        ActiveGrammarFst* active_grammar_fst = nullptr;

        // Decoder objects
        OnlineNnet2FeaturePipeline* feature_pipeline = nullptr;
        OnlineSilenceWeighting* silence_weighting = nullptr;  // reinstantiated per utterance
        OnlineIvectorExtractorAdaptationState* adaptation_state = nullptr;
        SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>* decoder = nullptr;
        WordAlignLatticeLexiconInfo* word_align_lexicon_info = nullptr;
        std::set<int32> word_align_lexicon_words;  // contains word-ids that are in word_align_lexicon_info
        std::vector<std::pair<CompactLattice::Arc::Label, CompactLattice::Arc::Label>> rule_relabel_ipairs_, rule_relabel_opairs_;

        int32 tot_frames = 0, tot_frames_decoded = 0;
        bool decoder_finalized_ = false;
        CompactLattice decoded_clat;
        CompactLattice best_path_clat;

        StdConstFst* ReadFstFile(std::string filename);
        void StartDecoding(std::vector<bool> grammars_activity);
        void CleanupDecoder();
        std::string WordIdsToString(const std::vector<int32> &wordIds);
};

AgfNNet3OnlineModelWrapper::AgfNNet3OnlineModelWrapper(const std::string& model_dir, const std::string& config_str, int32 verbosity) {
    SetVerboseLevel(verbosity);
    if (verbosity >= 0) {
        KALDI_LOG << "model_dir: " << model_dir;
        KALDI_LOG << "config_str: " << config_str;
        KALDI_LOG << "verbosity: " << verbosity;
    } else if (verbosity == -1) {
        SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {
            if (envelope.severity <= LogMessageEnvelope::kWarning) {
                std::cerr << "[KALDI severity=" << envelope.severity << "] " << message << "\n";
            }
        });
    } else {
        // Silence kaldi output as well
        SetLogHandler([](const LogMessageEnvelope& envelope, const char* message) {});
    }

    config_.model_dir = model_dir;
    if (!config_str.empty()) {
        auto config_json = nlohmann::json::parse(config_str);
        if (!config_json.is_object())
            KALDI_ERR << "config_str must be a valid JSON object";
        for (const auto& it : config_json.items()) {
            if (!config_.Set(it.key(), it.value()))
                KALDI_WARN << "Bad config key: " << it.key() << " = " << it.value();
        }
    }
    KALDI_LOG << config_.ToString();

    if (true && verbosity >= 1) {
        ExecutionTimer timer("testing output latency");
        std::cerr << "[testing output latency][testing output latency][testing output latency]" << endl;
    }

    ParseOptions po("");
    feature_config.Register(&po);
    decodable_config.Register(&po);
    decoder_config.Register(&po);
    endpoint_config.Register(&po);

    feature_config.mfcc_config = config_.mfcc_config_filename;
    feature_config.ivector_extraction_config = config_.ie_config_filename;
    feature_config.silence_weighting_config.silence_weight = config_.silence_weight;
    feature_config.silence_weighting_config.silence_phones_str = config_.silence_phones_str;
    decoder_config.max_active = config_.max_active;
    decoder_config.min_active = config_.min_active;
    decoder_config.beam = config_.beam;
    decoder_config.lattice_beam = config_.lattice_beam;
    decodable_config.acoustic_scale = config_.acoustic_scale;
    decodable_config.frame_subsampling_factor = config_.frame_subsampling_factor;

    KALDI_VLOG(2) << "kNontermBigNumber, GetEncodingMultiple: " << kNontermBigNumber << ", " << GetEncodingMultiple(config_.nonterm_phones_offset);

    {
        bool binary;
        Input ki(config_.model_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_nnet.Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
        SetDropoutTestMode(true, &(am_nnet.GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    feature_info = new OnlineNnet2FeaturePipelineInfo(feature_config);
    decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_config, &am_nnet);
    ResetAdaptationState();
    top_fst = dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(config_.top_fst_filename));

    if (!config_.dictation_fst_filename.empty())
        dictation_fst = ReadFstFile(config_.dictation_fst_filename);

    LoadLexicon(config_.word_syms_filename, config_.word_align_lexicon_filename);

    auto first_rule_sym = word_syms->Find("#nonterm:rule0"),
        last_rule_sym = first_rule_sym + 9999;
    for (auto i = first_rule_sym; i <= last_rule_sym; ++i)
        rule_relabel_ipairs_.emplace_back(std::make_pair(i, first_rule_sym));
}

AgfNNet3OnlineModelWrapper::~AgfNNet3OnlineModelWrapper() {
    CleanupDecoder();
    delete feature_info;
    delete decodable_info;
    if (word_align_lexicon_info)
        delete word_align_lexicon_info;
}

bool AgfNNet3OnlineModelWrapper::LoadLexicon(std::string& word_syms_filename, std::string& word_align_lexicon_filename) {
    // FIXME: make more robust to errors

    if (word_syms_filename != "") {
        if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
            KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
            return false;
        }
    }

    if (word_align_lexicon_filename != "") {
        bool binary_in;
        Input ki(word_align_lexicon_filename, &binary_in);
        KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
        if (!ReadLexiconForWordAlign(ki.Stream(), &word_align_lexicon)) {
            KALDI_ERR << "Error reading word alignment lexicon from file " << word_align_lexicon_filename;
            return false;
        }
        if (word_align_lexicon_info)
            delete word_align_lexicon_info;
        word_align_lexicon_info = new WordAlignLatticeLexiconInfo(word_align_lexicon);

        word_align_lexicon_words.clear();
        for (auto entry : word_align_lexicon)
            word_align_lexicon_words.insert(entry.at(0));
    }

    return true;
}

StdConstFst* AgfNNet3OnlineModelWrapper::ReadFstFile(std::string filename) {
    if (filename.compare(filename.length() - 4, 4, ".txt") == 0) {
        // FIXME: fstdeterminize | fstminimize | fstrmepsilon | fstarcsort --sort_type=ilabel
        KALDI_WARN << "cannot read text fst file " << filename;
        return nullptr;
    } else {
        return dynamic_cast<StdConstFst*>(ReadFstKaldiGeneric(filename));
    }
}

int32 AgfNNet3OnlineModelWrapper::AddGrammarFst(std::string& grammar_fst_filename) {
    auto grammar_fst_index = grammar_fsts.size();
    auto grammar_fst = ReadFstFile(grammar_fst_filename);
    KALDI_VLOG(2) << "adding FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst_filename;
    grammar_fsts.emplace_back(grammar_fst);
    grammar_fsts_filename_map[grammar_fst] = grammar_fst_filename;
    if (active_grammar_fst) {
        delete active_grammar_fst;
        active_grammar_fst = nullptr;
    }
    return grammar_fst_index;
}

bool AgfNNet3OnlineModelWrapper::ReloadGrammarFst(int32 grammar_fst_index, std::string& grammar_fst_filename) {
    auto old_grammar_fst = grammar_fsts.at(grammar_fst_index);
    grammar_fsts_filename_map.erase(old_grammar_fst);
    delete old_grammar_fst;

    auto grammar_fst = ReadFstFile(grammar_fst_filename);
    KALDI_VLOG(2) << "reloading FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fst_filename;
    grammar_fsts.at(grammar_fst_index) = grammar_fst;
    grammar_fsts_filename_map[grammar_fst] = grammar_fst_filename;
    if (active_grammar_fst) {
        delete active_grammar_fst;
        active_grammar_fst = nullptr;
    }
    return true;
}

bool AgfNNet3OnlineModelWrapper::RemoveGrammarFst(int32 grammar_fst_index) {
    auto grammar_fst = grammar_fsts.at(grammar_fst_index);
    KALDI_VLOG(2) << "removing FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_fsts_filename_map.at(grammar_fst);
    grammar_fsts.erase(grammar_fsts.begin() + grammar_fst_index);
    grammar_fsts_filename_map.erase(grammar_fst);
    delete grammar_fst;
    if (active_grammar_fst) {
        delete active_grammar_fst;
        active_grammar_fst = nullptr;
    }
    return true;
}

bool AgfNNet3OnlineModelWrapper::SaveAdaptationState() {
    if (feature_pipeline != nullptr) {
        feature_pipeline->GetAdaptationState(adaptation_state);
        KALDI_LOG << "Saved adaptation state.";
        return true;
    }
    return false;
}

void AgfNNet3OnlineModelWrapper::ResetAdaptationState() {
    if (adaptation_state != nullptr) {
        delete adaptation_state;
    }
    adaptation_state = new OnlineIvectorExtractorAdaptationState(feature_info->ivector_extractor_info);
}

void AgfNNet3OnlineModelWrapper::StartDecoding(std::vector<bool> grammars_activity) {
    CleanupDecoder();
    ExecutionTimer timer("StartDecoding", 2);

    if (active_grammar_fst == nullptr) {
        std::vector<std::pair<int32, const StdConstFst *> > ifsts;
        for (auto grammar_fst : grammar_fsts) {
            int32 nonterm_phone = config_.rules_phones_offset + ifsts.size();
            ifsts.emplace_back(std::make_pair(nonterm_phone, grammar_fst));
        }
        if (dictation_fst != nullptr) {
            ifsts.emplace_back(std::make_pair(config_.dictation_phones_offset, dictation_fst));
        }
        active_grammar_fst = new ActiveGrammarFst(config_.nonterm_phones_offset, *top_fst, ifsts);
    }
    grammars_activity.push_back(dictation_fst != nullptr);  // dictation_fst is only enabled if present
    active_grammar_fst->UpdateActivity(grammars_activity);

    feature_pipeline = new OnlineNnet2FeaturePipeline(*feature_info);
    feature_pipeline->SetAdaptationState(*adaptation_state);
    silence_weighting = new OnlineSilenceWeighting(
        trans_model, feature_info->silence_weighting_config,
        decodable_config.frame_subsampling_factor);
    decoder = new SingleUtteranceNnet3DecoderTpl<fst::ActiveGrammarFst>(
        decoder_config, trans_model, *decodable_info, *active_grammar_fst, feature_pipeline);

    // Cleanup
    decoder_finalized_ = false;
    decoded_clat.DeleteStates();
    best_path_clat.DeleteStates();
}

void AgfNNet3OnlineModelWrapper::CleanupDecoder() {
    if (decoder) {
        delete decoder;
        decoder = nullptr;
    }
    if (silence_weighting) {
        delete silence_weighting;
        silence_weighting = nullptr;
    }
    if (feature_pipeline) {
        delete feature_pipeline;
        feature_pipeline = nullptr;
    }
}

// grammars_activity is ignored once decoding has already started
bool AgfNNet3OnlineModelWrapper::Decode(BaseFloat samp_freq, const Vector<BaseFloat>& samples, bool finalize,
        std::vector<bool>& grammars_activity, bool save_adaptation_state) {
    ExecutionTimer timer("Decode", 2);

    if (!decoder || decoder_finalized_) {
        CleanupDecoder();
        StartDecoding(grammars_activity);
    } else if (grammars_activity.size() != 0) {
    	KALDI_LOG << "non-empty grammars_activity passed on already-started decode";
    }

    if (samp_freq != feature_info->GetSamplingFrequency())
        KALDI_WARN << "Mismatched sampling frequency: " << samp_freq << " != " << feature_info->GetSamplingFrequency() << " (model's)";

    if (samples.Dim() > 0) {
        feature_pipeline->AcceptWaveform(samp_freq, samples);
        tot_frames += samples.Dim();
    }

    if (finalize)
        feature_pipeline->InputFinished();  // No more input, so flush out last frames.

    if (silence_weighting->Active()
            && feature_pipeline->NumFramesReady() > 0
            && feature_pipeline->IvectorFeature() != nullptr) {
        if (config_.silence_weight == 1.0)
            KALDI_WARN << "Computing silence weighting despite silence_weight == 1.0";
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
        silence_weighting->ComputeCurrentTraceback(decoder->Decoder());
        silence_weighting->GetDeltaWeights(feature_pipeline->NumFramesReady(), &delta_weights);  // FIXME: reuse decoder?
        feature_pipeline->IvectorFeature()->UpdateFrameWeights(delta_weights);
    }

    decoder->AdvanceDecoding();

    if (finalize) {
        ExecutionTimer timer("Decode finalize", 2);
        decoder->FinalizeDecoding();
        decoder_finalized_ = true;

        tot_frames_decoded += tot_frames;
        tot_frames = 0;

        if (save_adaptation_state) {
            feature_pipeline->GetAdaptationState(adaptation_state);
            KALDI_LOG << "Saved adaptation state";
            // std::string output;
            // double likelihood;
            // GetDecodedString(output, likelihood);
            // // int count_terminals = std::count_if(output.begin(), output.end(), [](std::string word){ return word[0] != '#'; });
            // if (output.size() > 0) {
            //     feature_pipeline->GetAdaptationState(adaptation_state);
            //     KALDI_LOG << "Saved adaptation state." << output;
            //     free_decoder();
            // } else {
            //     KALDI_LOG << "Did not save adaptation state, because empty recognition.";
            // }
        }
    }

    return true;
}

void AgfNNet3OnlineModelWrapper::GetDecodedString(std::string& decoded_string, float* likelihood, float* am_score, float* lm_score, float* confidence, float* expected_error_rate) {
    ExecutionTimer timer("GetDecodedString", 2);

    decoded_string = "";
    if (likelihood) *likelihood = NAN;
    if (confidence) *confidence = NAN;
    if (expected_error_rate) *expected_error_rate = NAN;
    if (lm_score) *lm_score = NAN;
    if (am_score) *am_score = NAN;

    if (!decoder) KALDI_ERR << "No decoder";
    if (decoder->NumFramesDecoded() == 0) {
        if (decoder_finalized_) KALDI_WARN << "GetDecodedString on empty decoder";
        // else KALDI_VLOG(2) << "GetDecodedString on empty decoder";
        return;
    }

    Lattice best_path_lat;
    if (!decoder_finalized_) {
        // Decoding is not finished yet, so we will just look up the best partial result so far
        decoder->GetBestPath(false, &best_path_lat);

    } else {
        decoder->GetLattice(true, &decoded_clat);
        if (decoded_clat.NumStates() == 0) KALDI_ERR << "Empty decoded lattice";
        ScaleLattice(LatticeScale(config_.lm_weight, 10.0), &decoded_clat);
        // WriteLattice(decoded_clat, "tmp/lattice");

        // FIXME
        // BaseFloat inv_acoustic_scale = 1.0 / decodable_config.acoustic_scale;
        // ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &decoded_clat);

        CompactLattice decoded_clat_relabeled = decoded_clat;
        if (true) {
            // Relabel all nonterm:rules to nonterm:rule0, so redundant/ambiguous rules don't count as differing for measuring confidence
            ExecutionTimer timer("relabel");
            Relabel(&decoded_clat_relabeled, rule_relabel_ipairs_, rule_relabel_opairs_);
            // FIXME: write a custom ArcMapper to do relabeling faster by checking if isym is in the range of nonterm:rules?
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
            auto nonterm_dictation = word_syms->Find("#nonterm:dictation");
            auto nonterm_end = word_syms->Find("#nonterm:end");
            bool ok;
            CopyDictationVisitor<CompactLatticeArc> visitor(&pre_dictation_clat, &in_dictation_clat, &post_dictation_clat, &ok, nonterm_dictation, nonterm_end);
            KALDI_ASSERT(ok);
            AnyArcFilter<CompactLatticeArc> filter;
            DfsVisit(decoded_clat, &visitor, filter, true);
            WriteLattice(in_dictation_clat, "tmp/lattice_dict");
            WriteLattice(pre_dictation_clat, "tmp/lattice_dictpre");
            WriteLattice(post_dictation_clat, "tmp/lattice_dictpost");
        }

        CompactLatticeShortestPath(decoded_clat, &best_path_clat);
        ConvertLattice(best_path_clat, &best_path_lat);
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

std::string AgfNNet3OnlineModelWrapper::WordIdsToString(const std::vector<int32> &wordIds) {
    stringstream text;
    for (size_t i = 0; i < wordIds.size(); i++) {
        std::string s = word_syms->Find(wordIds[i]);
        if (s == "") {
            KALDI_WARN << "Word-id " << wordIds[i] << " not in symbol table";
            s = "MISSING_WORD";
        }
        if (i != 0) text << " ";
        text << word_syms->Find(wordIds[i]);
    }
    return text.str();
}

bool AgfNNet3OnlineModelWrapper::GetWordAlignment(std::vector<string>& words, std::vector<int32>& times, std::vector<int32>& lengths, bool include_eps) {
    if (!word_align_lexicon.size() || !word_align_lexicon_info) KALDI_ERR << "No word alignment lexicon loaded";
    if (best_path_clat.NumStates() == 0) KALDI_ERR << "No best path lattice";

    // if (!best_path_has_valid_word_align) {
    //     KALDI_ERR << "There was a word not in word alignment lexicon";
    // }
    // if (!word_align_lexicon_words.count(words[i])) {
    //     KALDI_LOG << "Word " << s << " (id #" << words[i] << ") not in word alignment lexicon";
    // }

    CompactLattice aligned_clat;
    WordAlignLatticeLexiconOpts opts;
    bool ok = WordAlignLatticeLexicon(best_path_clat, trans_model, *word_align_lexicon_info, opts, &aligned_clat);

    if (!ok) {
        KALDI_WARN << "Lattice did not align correctly";
        return false;
    }

    if (aligned_clat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Lattice was empty";
        return false;
    }

    TopSortCompactLatticeIfNeeded(&aligned_clat);

    // lattice-1best
    CompactLattice best_path_aligned;
    CompactLatticeShortestPath(aligned_clat, &best_path_aligned);

    // nbest-to-ctm
    std::vector<int32> word_idxs, times_raw, lengths_raw;
    ok = CompactLatticeToWordAlignment(best_path_aligned, &word_idxs, &times_raw, &lengths_raw);
    if (!ok) {
        KALDI_WARN << "CompactLatticeToWordAlignment failed.";
        return false;
    }

    // lexicon lookup
    words.clear();
    for (size_t i = 0; i < word_idxs.size(); i++) {
        std::string s = word_syms->Find(word_idxs[i]);  // Must be found, or CompactLatticeToWordAlignment would have crashed
        // KALDI_LOG << "align: " << s << " - " << times_raw[i] << " - " << lengths_raw[i];
        if (include_eps || (word_idxs[i] != 0)) {
            words.push_back(s);
            times.push_back(times_raw[i]);
            lengths.push_back(lengths_raw[i]);
        }
    }
    return true;
}

}  // namespace dragonfly


using namespace dragonfly;

void* init_agf_nnet3(char* model_dir_cp, char* config_str_cp, int32_t verbosity) {
    std::string model_dir(model_dir_cp),
        config_str((config_str_cp != nullptr) ? config_str_cp : "");
    AgfNNet3OnlineModelWrapper* model = new AgfNNet3OnlineModelWrapper(model_dir, config_str, verbosity);
    return model;
}

bool load_lexicon_agf_nnet3(void* model_vp, char* word_syms_filename_cp, char* word_align_lexicon_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string word_syms_filename(word_syms_filename_cp), word_align_lexicon_filename(word_align_lexicon_filename_cp);
    bool result = model->LoadLexicon(word_syms_filename, word_align_lexicon_filename);
    return result;
}

int32_t add_grammar_fst_agf_nnet3(void* model_vp, char* grammar_fst_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    int32_t grammar_fst_index = model->AddGrammarFst(grammar_fst_filename);
    return grammar_fst_index;
}

bool reload_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index, char* grammar_fst_filename_cp) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    std::string grammar_fst_filename(grammar_fst_filename_cp);
    bool result = model->ReloadGrammarFst(grammar_fst_index, grammar_fst_filename);
    return result;
}

bool remove_grammar_fst_agf_nnet3(void* model_vp, int32_t grammar_fst_index) {
    AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
    bool result = model->RemoveGrammarFst(grammar_fst_index);
    return result;
}

bool decode_agf_nnet3(void* model_vp, float samp_freq, int32_t num_samples, float* samples, bool finalize,
    bool* grammars_activity_cp, int32_t grammars_activity_cp_size, bool save_adaptation_state) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::vector<bool> grammars_activity(grammars_activity_cp_size, false);
        for (size_t i = 0; i < grammars_activity_cp_size; i++)
            grammars_activity[i] = grammars_activity_cp[i];
        // if (num_samples > 3200)
        //     KALDI_WARN << "Decoding large block of " << num_samples << " samples!";
        Vector<BaseFloat> wave_data(num_samples, kUndefined);
        for (int i = 0; i < num_samples; i++)
            wave_data(i) = samples[i];
        bool result = model->Decode(samp_freq, wave_data, finalize, grammars_activity, save_adaptation_state);
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool save_adaptation_state_agf_nnet3(void* model_vp) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        bool result = model->SaveAdaptationState();
        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool reset_adaptation_state_agf_nnet3(void* model_vp) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        model->ResetAdaptationState();
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_output_agf_nnet3(void* model_vp, char* output, int32_t output_max_length,
        float* likelihood_p, float* am_score_p, float* lm_score_p, float* confidence_p, float* expected_error_rate_p) {
    try {
        if (output_max_length < 1) return false;
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::string decoded_string;
	    model->GetDecodedString(decoded_string, likelihood_p, am_score_p, lm_score_p, confidence_p, expected_error_rate_p);

        // KALDI_LOG << "sleeping";
        // std::this_thread::sleep_for(std::chrono::milliseconds(25));
        // KALDI_LOG << "slept";

        const char* cstr = decoded_string.c_str();
        strncpy(output, cstr, output_max_length);
        output[output_max_length - 1] = 0;
        return true;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}

bool get_word_align_agf_nnet3(void* model_vp, int32_t* times_cp, int32_t* lengths_cp, int32_t num_words) {
    try {
        AgfNNet3OnlineModelWrapper* model = static_cast<AgfNNet3OnlineModelWrapper*>(model_vp);
        std::vector<string> words;
        std::vector<int32> times, lengths;
        bool result = model->GetWordAlignment(words, times, lengths, false);

        if (result) {
            KALDI_ASSERT(words.size() == num_words);
            for (size_t i = 0; i < words.size(); i++) {
                times_cp[i] = times[i];
                lengths_cp[i] = lengths[i];
            }
        } else {
            KALDI_WARN << "alignment failed";
        }

        return result;

    } catch(const std::exception& e) {
        KALDI_WARN << "Trying to survive fatal exception: " << e.what();
        return false;
    }
}
