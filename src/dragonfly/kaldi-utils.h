// Kaldi Utils for AG

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

#pragma once

#include <ctime>
#include <iomanip>
#include "fstext/fstext-lib.h"
#include "online2/online-ivector-feature.h"

#include "nlohmann_json.hpp"

namespace kaldi {
    void from_json(const nlohmann::json& j, OnlineIvectorExtractionConfig& c);
}

namespace dragonfly {

using namespace kaldi;
using namespace fst;


inline ConstFst<StdArc>* CastOrConvertToConstFst(Fst<StdArc>* fst, bool has_ownership = true) {
    // This version currently supports ConstFst<StdArc> or VectorFst<StdArc>
    std::string real_type = fst->Type();
    KALDI_ASSERT(real_type == "vector" || real_type == "const");
    if (real_type == "const") {
        return dynamic_cast<ConstFst<StdArc>*>(fst);
    } else {
        // As the 'fst' can't cast to ConstFst, we carete a new
        // ConstFst<StdArc> initialized by 'fst', and delete 'fst'.
        ConstFst<StdArc>* new_fst = new ConstFst<StdArc>(*fst);
        if (has_ownership) {
            delete fst;
        }
        return new_fst;
    }
}

inline void WriteLattice(const CompactLattice clat_in, std::string name = "lattice") {
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


// RAII object that sets Kaldi verbosity level upon construction, and resets it upon destruction.
class VerboseLevelResetter {
   public:
    VerboseLevelResetter(int32 verbosity) {
        orig_verbosity_ = GetVerboseLevel();
        SetVerboseLevel(verbosity);
    }

    ~VerboseLevelResetter() {
        SetVerboseLevel(orig_verbosity_);
    }

    int32 GetOrigVerbosity() const { return orig_verbosity_; }

   private:
    int32 orig_verbosity_;
};


// ArcMapper that takes acceptor and relabels all nonterm:rules to nonterm:rule0, so redundant/ambiguous rules don't count as differing for measuring
// confidence.
template <class A>
class CombineRuleNontermMapper {
   public:
    using FromArc = A;
    using ToArc = A;
    using Label = typename FromArc::Label;

    explicit CombineRuleNontermMapper(Label first_rule_sym, Label last_rule_sym) : first_rule_sym_(first_rule_sym), last_rule_sym_(last_rule_sym) {}

    ToArc operator()(const FromArc& arc) const {
        // KALDI_ASSERT(arc.ilabel == arc.olabel);
        if ((arc.ilabel >= first_rule_sym_) && (arc.ilabel <= last_rule_sym_))
            return ToArc(first_rule_sym_, first_rule_sym_, arc.weight, arc.nextstate);
        else
            return arc;
    }

    constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

    constexpr MapSymbolsAction InputSymbolsAction() const { return MAP_NOOP_SYMBOLS; }

    constexpr MapSymbolsAction OutputSymbolsAction() const { return MAP_NOOP_SYMBOLS; }

    uint64 Properties(uint64 props) const {
        KALDI_ASSERT(props & kAcceptor);
        return (props & kSetArcProperties);
    }

   private:
    Label first_rule_sym_;
    Label last_rule_sym_;
};

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

} // namespace dragonfly
