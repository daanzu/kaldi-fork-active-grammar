// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes for the recursive replacement of FSTs.

#ifndef FST_ACTIVE_REPLACE_H_
#define FST_ACTIVE_REPLACE_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/expanded-fst.h>
#include <fst/fst-decl.h>  // For optional argument declarations.
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/replace-util.h>
#include <fst/state-table.h>
#include <fst/test-properties.h>

#include "active-cache.h"

namespace fst {

// By default ActiveReplaceFst will copy the input label of the replace arc.
// The call_label_type and return_label_type options specify how to manage
// the labels of the call arc and the return arc of the replace FST
template <class Arc, class StateTable = DefaultReplaceStateTable<Arc>,
          class CacheStore = DefaultActiveCacheStore<Arc>>
struct ActiveReplaceFstOptions : CacheImplOptions<CacheStore> {
  using Label = typename Arc::Label;

  // Index of root rule for expansion.
  Label root;
  // How to label call arc.
  ReplaceLabelType call_label_type = REPLACE_LABEL_INPUT;
  // How to label return arc.
  ReplaceLabelType return_label_type = REPLACE_LABEL_NEITHER;
  // Specifies output label to put on call arc; if kNoLabel, use existing label
  // on call arc. Otherwise, use this field as the output label.
  Label call_output_label = kNoLabel;
  // Specifies label to put on return arc.
  Label return_label = 0;
  // Take ownership of input FSTs?
  bool take_ownership = false;
  // Pointer to optional pre-constructed state table.
  StateTable *state_table = nullptr;

  explicit ActiveReplaceFstOptions(const CacheImplOptions<CacheStore> &opts,
                             Label root = kNoLabel)
      : CacheImplOptions<CacheStore>(opts), root(root) {}

  explicit ActiveReplaceFstOptions(const CacheOptions &opts, Label root = kNoLabel)
      : CacheImplOptions<CacheStore>(opts), root(root) {}

  // FIXME(kbg): There are too many constructors here. Come up with a consistent
  // position for call_output_label (probably the very end) so that it is
  // possible to express all the remaining constructors with a single
  // default-argument constructor. Also move clients off of the "backwards
  // compatibility" constructor, for good.

  explicit ActiveReplaceFstOptions(Label root) : root(root) {}

  explicit ActiveReplaceFstOptions(Label root, ReplaceLabelType call_label_type,
                             ReplaceLabelType return_label_type,
                             Label return_label)
      : root(root),
        call_label_type(call_label_type),
        return_label_type(return_label_type),
        return_label(return_label) {}

  explicit ActiveReplaceFstOptions(Label root, ReplaceLabelType call_label_type,
                             ReplaceLabelType return_label_type,
                             Label call_output_label, Label return_label)
      : root(root),
        call_label_type(call_label_type),
        return_label_type(return_label_type),
        call_output_label(call_output_label),
        return_label(return_label) {}

  explicit ActiveReplaceFstOptions(const ReplaceUtilOptions &opts)
      : ActiveReplaceFstOptions(opts.root, opts.call_label_type,
                          opts.return_label_type, opts.return_label) {}

  ActiveReplaceFstOptions() : root(kNoLabel) {}

  // For backwards compatibility.
  ActiveReplaceFstOptions(int64 root, bool epsilon_replace_arc)
      : root(root),
        call_label_type(epsilon_replace_arc ? REPLACE_LABEL_NEITHER
                                            : REPLACE_LABEL_INPUT),
        call_output_label(epsilon_replace_arc ? 0 : kNoLabel) {}
};


// Forward declaration.
template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFstMatcher;

template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFst;

template <class Arc>
using FstList = std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>;

namespace internal {

// The replace implementation class supports a dynamic expansion of a recursive
// transition network represented as label/FST pairs with dynamic replacable
// arcs.
template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFstImpl
    : public CacheBaseImpl<typename CacheStore::State, CacheStore> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using State = typename CacheStore::State;
  using CacheImpl = CacheBaseImpl<State, CacheStore>;
  using PrefixId = typename StateTable::PrefixId;
  using StateTuple = ReplaceStateTuple<StateId, PrefixId>;
  using StackPrefix = ReplaceStackPrefix<Label, StateId>;
  using NonTerminalHash = std::unordered_map<Label, Label>;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::WriteHeader;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::InputSymbols;
  using FstImpl<Arc>::OutputSymbols;

  using CacheImpl::PushArc;
  using CacheImpl::HasArcs;
  using CacheImpl::HasFinal;
  using CacheImpl::HasStart;
  using CacheImpl::SetArcs;
  using CacheImpl::SetFinal;
  using CacheImpl::SetStart;

  friend class ActiveReplaceFst<Arc, StateTable, CacheStore>;
  friend class ActiveReplaceFstMatcher<Arc, StateTable, CacheStore>;

  ActiveReplaceFstImpl(const FstList<Arc> &fst_list,
                 const ActiveReplaceFstOptions<Arc, StateTable, CacheStore> &opts)
      : CacheImpl(opts),
        call_label_type_(opts.call_label_type),
        return_label_type_(opts.return_label_type),
        call_output_label_(opts.call_output_label),
        return_label_(opts.return_label),
        state_table_(opts.state_table ? opts.state_table
                                      : new StateTable(fst_list, opts.root)) {
    SetType("active-replace");
    // If the label is epsilon, then all replace label options are equivalent,
    // so we set the label types to NEITHER for simplicity.
    if (call_output_label_ == 0) call_label_type_ = REPLACE_LABEL_NEITHER;
    if (return_label_ == 0) return_label_type_ = REPLACE_LABEL_NEITHER;
    if (!fst_list.empty()) {
      SetInputSymbols(fst_list[0].second->InputSymbols());
      SetOutputSymbols(fst_list[0].second->OutputSymbols());
    }
    fst_array_.push_back(nullptr);
    for (Label i = 0; i < fst_list.size(); ++i) {
      const auto label = fst_list[i].first;
      const auto *fst = fst_list[i].second;
      nonterminal_hash_[label] = fst_array_.size();
      nonterminal_set_.insert(label);
      fst_array_.emplace_back(opts.take_ownership ? fst : fst->Copy());
      if (i) {
        if (!CompatSymbols(InputSymbols(), fst->InputSymbols())) {
          FSTERROR() << "ActiveReplaceFstImpl: Input symbols of FST " << i
                     << " do not match input symbols of base FST (0th FST)";
          SetProperties(kError, kError);
        }
        if (!CompatSymbols(OutputSymbols(), fst->OutputSymbols())) {
          FSTERROR() << "ActiveReplaceFstImpl: Output symbols of FST " << i
                     << " do not match output symbols of base FST (0th FST)";
          SetProperties(kError, kError);
        }
      }
    }

    if (!nonterminal_set_.empty()) CacheImpl::GetCacheStore()->SetNonterminals(*nonterminal_set_.cbegin(), *nonterminal_set_.crbegin());
    fst_activity_.assign(fst_array_.size(), false);

    const auto nonterminal = nonterminal_hash_[opts.root];
    if ((nonterminal == 0) && (fst_array_.size() > 1)) {
      FSTERROR() << "ActiveReplaceFstImpl: No FST corresponding to root label "
                 << opts.root << " in the input tuple vector";
      SetProperties(kError, kError);
    }
    root_ = (nonterminal > 0) ? nonterminal : 1;
    bool all_non_empty_and_sorted = false;
    SetProperties(ReplaceFstProperties(opts.root, fst_list, call_label_type_,
                                       return_label_type_, call_output_label_,
                                       &all_non_empty_and_sorted));
    // Enables optional caching as long as sorted and all non-empty.
    always_cache_ = !all_non_empty_and_sorted;
    VLOG(2) << "ActiveReplaceFstImpl::ActiveReplaceFstImpl: always_cache = "
            << (always_cache_ ? "true" : "false");
  }

  ActiveReplaceFstImpl(const ActiveReplaceFstImpl &impl)
      : CacheImpl(impl),
        call_label_type_(impl.call_label_type_),
        return_label_type_(impl.return_label_type_),
        call_output_label_(impl.call_output_label_),
        return_label_(impl.return_label_),
        always_cache_(impl.always_cache_),
        state_table_(new StateTable(*(impl.state_table_))),
        nonterminal_set_(impl.nonterminal_set_),
        nonterminal_hash_(impl.nonterminal_hash_),
        fst_activity_(impl.fst_activity_),
        root_(impl.root_) {
    SetType("replace");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
    fst_array_.reserve(impl.fst_array_.size());
    fst_array_.emplace_back(nullptr);
    for (Label i = 1; i < impl.fst_array_.size(); ++i) {
      fst_array_.emplace_back(impl.fst_array_[i]->Copy(true));
    }
    if (!nonterminal_set_.empty()) CacheImpl::GetCacheStore()->SetNonterminals(*nonterminal_set_.cbegin(), *nonterminal_set_.crbegin());
  }

  // Computes the dependency graph of the replace class and returns
  // true if the dependencies are cyclic. Cyclic dependencies will result
  // in an un-expandable FST.
  bool CyclicDependencies() const {
    const ReplaceUtilOptions opts(root_);
    ReplaceUtil<Arc> replace_util(fst_array_, nonterminal_hash_, opts);
    return replace_util.CyclicDependencies();
  }

  StateId Start() {
    if (!HasStart()) {
      if (fst_array_.size() == 1) {
        SetStart(kNoStateId);
        return kNoStateId;
      } else {
        const auto fst_start = fst_array_[root_]->Start();
        if (fst_start == kNoStateId) return kNoStateId;
        const auto prefix = GetPrefixId(StackPrefix());
        const auto start =
            state_table_->FindState(StateTuple(prefix, root_, fst_start));
        SetStart(start);
        return start;
      }
    } else {
      return CacheImpl::Start();
    }
  }

  Weight Final(StateId s) {
    if (HasFinal(s)) return CacheImpl::Final(s);
    const auto &tuple = state_table_->Tuple(s);
    auto weight = Weight::Zero();
    if (tuple.prefix_id == 0) {
      const auto fst_state = tuple.fst_state;
      weight = fst_array_[tuple.fst_id]->Final(fst_state);
    }
    if (always_cache_ || HasArcs(s)) SetFinal(s, weight);
    return weight;
  }

  size_t NumArcs(StateId s) {
    if (HasArcs(s)) {
      return CacheImpl::NumArcs(s);
    } else if (always_cache_) {  // If always caching, expands and caches state.
      Expand(s);
      return CacheImpl::NumArcs(s);
    } else {  // Otherwise computes the number of arcs without expanding.
      const auto tuple = state_table_->Tuple(s);
      if (tuple.fst_state == kNoStateId) return 0;
      auto num_arcs = fst_array_[tuple.fst_id]->NumArcs(tuple.fst_state);
      if (ComputeFinalArc(tuple, nullptr)) ++num_arcs;
      return num_arcs;
    }
  }

  // Returns whether a given label is a non-terminal.
  bool IsNonTerminal(Label label) const {
    if (label < *nonterminal_set_.begin() ||
        label > *nonterminal_set_.rbegin()) {
      return false;
    } else {
      return true;  // ActiveReplaceFst!
      // return nonterminal_hash_.count(label);
    }
    // TODO(allauzen): be smarter and take advantage of all_dense or
    // all_negative. Also use this in ComputeArc. This would require changes to
    // Replace so that recursing into an empty FST lead to a non co-accessible
    // state instead of deleting the arc as done currently. The current use
    // correct, since labels are sorted if all_non_empty is true.
  }

  size_t NumInputEpsilons(StateId s) {
    if (HasArcs(s)) {
      return CacheImpl::NumInputEpsilons(s);
    } else if (always_cache_ || !Properties(kILabelSorted)) {
      // If always caching or if the number of input epsilons is too expensive
      // to compute without caching (i.e., not ilabel-sorted), then expands and
      // caches state.
      Expand(s);
      return CacheImpl::NumInputEpsilons(s);
    } else {
      // Otherwise, computes the number of input epsilons without caching.
      const auto tuple = state_table_->Tuple(s);
      if (tuple.fst_state == kNoStateId) return 0;
      size_t num = 0;
      if (!EpsilonOnInput(call_label_type_)) {
        // If EpsilonOnInput(c) is false, all input epsilon arcs
        // are also input epsilons arcs in the underlying machine.
        num = fst_array_[tuple.fst_id]->NumInputEpsilons(tuple.fst_state);
      } else {
        // Otherwise, one need to consider that all non-terminal arcs
        // in the underlying machine also become input epsilon arc.
        ArcIterator<Fst<Arc>> aiter(*fst_array_[tuple.fst_id], tuple.fst_state);
        for (; !aiter.Done() && ((aiter.Value().ilabel == 0) ||
                                 IsNonTerminal(aiter.Value().olabel));
             aiter.Next()) {
          ++num;
        }
      }
      if (EpsilonOnInput(return_label_type_) &&
          ComputeFinalArc(tuple, nullptr)) {
        ++num;
      }
      return num;
    }
  }

  size_t NumOutputEpsilons(StateId s) {
    if (HasArcs(s)) {
      return CacheImpl::NumOutputEpsilons(s);
    } else if (always_cache_ || !Properties(kOLabelSorted)) {
      // If always caching or if the number of output epsilons is too expensive
      // to compute without caching (i.e., not olabel-sorted), then expands and
      // caches state.
      Expand(s);
      return CacheImpl::NumOutputEpsilons(s);
    } else {
      // Otherwise, computes the number of output epsilons without caching.
      const auto tuple = state_table_->Tuple(s);
      if (tuple.fst_state == kNoStateId) return 0;
      size_t num = 0;
      if (!EpsilonOnOutput(call_label_type_)) {
        // If EpsilonOnOutput(c) is false, all output epsilon arcs are also
        // output epsilons arcs in the underlying machine.
        num = fst_array_[tuple.fst_id]->NumOutputEpsilons(tuple.fst_state);
      } else {
        // Otherwise, one need to consider that all non-terminal arcs in the
        // underlying machine also become output epsilon arc.
        ArcIterator<Fst<Arc>> aiter(*fst_array_[tuple.fst_id], tuple.fst_state);
        for (; !aiter.Done() && ((aiter.Value().olabel == 0) ||
                                 IsNonTerminal(aiter.Value().olabel));
             aiter.Next()) {
          ++num;
        }
      }
      if (EpsilonOnOutput(return_label_type_) &&
          ComputeFinalArc(tuple, nullptr)) {
        ++num;
      }
      return num;
    }
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if (mask & kError) {
      for (Label i = 1; i < fst_array_.size(); ++i) {
        if (fst_array_[i]->Properties(kError, false)) {
          SetProperties(kError, kError);
        }
      }
    }
    return FstImpl<Arc>::Properties(mask);
  }

  // Returns the base arc iterator, and if arcs have not been computed yet,
  // extends and recurses for new arcs.
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl::InitArcIterator(s, data);
    // TODO(allauzen): Set behaviour of generic iterator.
    // Warning: ArcIterator<ActiveReplaceFst<A>>::InitCache() relies on current
    // behaviour.
  }

  // Extends current state (walk arcs one level deep).
  void Expand(StateId s) {
    const auto tuple = state_table_->Tuple(s);
    if (tuple.fst_state == kNoStateId) {  // Local FST is empty.
      SetArcs(s);
      return;
    }

    ArcIterator<Fst<Arc>> aiter(*fst_array_[tuple.fst_id], tuple.fst_state);
    Arc arc;
    // Creates a final arc when needed.
    if (ComputeFinalArc(tuple, &arc)) PushArc(s, std::move(arc));
    bool any_nonterminal = false;
    // Expands all arcs leaving the state.
    for (; !aiter.Done(); aiter.Next()) {
      bool is_nonterminal = false;
      if (ComputeArc(tuple, aiter.Value(), &arc, &is_nonterminal)) PushArc(s, std::move(arc));
      any_nonterminal = any_nonterminal || is_nonterminal;
    }
    SetArcs(s);

    if (any_nonterminal) { CacheImpl::GetCacheStore()->SetStateActiveVolatility(s, any_nonterminal); }
  }

  // Dead code
  void Expand(StateId s, const StateTuple &tuple,
              const ArcIteratorData<Arc> &data) {
    KALDI_ERR << "Expand got called?!";
    if (tuple.fst_state == kNoStateId) {  // Local FST is empty.
      SetArcs(s);
      return;
    }
    ArcIterator<Fst<Arc>> aiter(data);
    Arc arc;
    // Creates a final arc when needed.
    if (ComputeFinalArc(tuple, &arc)) AddArc(s, arc);
    // Expands all arcs leaving the state.
    for (; !aiter.Done(); aiter.Next()) {
      if (ComputeArc(tuple, aiter.Value(), &arc)) AddArc(s, arc);
    }
    SetArcs(s);
  }

  // If acpp is null, only returns true if a final arcp is required, but does
  // not actually compute it.
  bool ComputeFinalArc(const StateTuple &tuple, Arc *arcp,
                       uint32 flags = kArcValueFlags) {
    const auto fst_state = tuple.fst_state;
    if (fst_state == kNoStateId) return false;
    // If state is final, pops the stack.
    if (fst_array_[tuple.fst_id]->Final(fst_state) != Weight::Zero() &&
        tuple.prefix_id) {
      if (arcp) {
        arcp->ilabel = (EpsilonOnInput(return_label_type_)) ? 0 : return_label_;
        arcp->olabel =
            (EpsilonOnOutput(return_label_type_)) ? 0 : return_label_;
        if (flags & kArcNextStateValue) {
          const auto &stack = state_table_->GetStackPrefix(tuple.prefix_id);
          const auto prefix_id = PopPrefix(stack);
          const auto &top = stack.Top();
          arcp->nextstate = state_table_->FindState(
              StateTuple(prefix_id, top.fst_id, top.nextstate));
        }
        if (flags & kArcWeightValue) {
          arcp->weight = fst_array_[tuple.fst_id]->Final(fst_state);
        }
      }
      return true;
    } else {
      return false;
    }
  }

  // Computes an arc in the FST corresponding to one in the underlying machine.
  // Returns false if the underlying arc corresponds to no arc in the resulting
  // FST.
  bool ComputeArc(const StateTuple &tuple, const Arc &arc, Arc *arcp,
                  bool *is_nonterminal = nullptr,
                  uint32 flags = kArcValueFlags) {
    if (!EpsilonOnInput(call_label_type_) &&
        (flags == (flags & (kArcILabelValue | kArcWeightValue)))) {
      *arcp = arc;
      return true;
    }
    if (arc.olabel == 0 || arc.olabel < *nonterminal_set_.begin() ||
        arc.olabel > *nonterminal_set_.rbegin()) {  // Expands local FST.
      const auto nextstate =
          flags & kArcNextStateValue
              ? state_table_->FindState(
                    StateTuple(tuple.prefix_id, tuple.fst_id, arc.nextstate))
              : kNoStateId;
      *arcp = Arc(arc.ilabel, arc.olabel, arc.weight, nextstate);
    } else {
      // Checks for non-terminal.
      const auto it = nonterminal_hash_.find(arc.olabel);
      if (it != nonterminal_hash_.end()) {  // Recurses into non-terminal.
        // FIXME: this side of the above branch path should be guaranteed for ActiveReplaceFst!
        if (is_nonterminal) {
          *is_nonterminal = true;
        }
        const auto nt_fst_id = it->second;
        // Drop arcs to inactive FSTs.
        if (!fst_activity_[nt_fst_id]) {
          return false;
        }
        const auto nt_prefix =
            PushPrefix(state_table_->GetStackPrefix(tuple.prefix_id),
                       tuple.fst_id, arc.nextstate);
        // If the start state is valid, replace; othewise, the arc is implicitly
        // deleted.
        const auto nt_start = fst_array_[nt_fst_id]->Start();
        if (nt_start != kNoStateId) {
          const auto nt_nextstate = flags & kArcNextStateValue
                                        ? state_table_->FindState(StateTuple(
                                              nt_prefix, nt_fst_id, nt_start))
                                        : kNoStateId;
          const auto ilabel =
              (EpsilonOnInput(call_label_type_)) ? 0 : arc.ilabel;
          const auto olabel =
              (EpsilonOnOutput(call_label_type_))
                  ? 0
                  : ((call_output_label_ == kNoLabel) ? arc.olabel
                                                      : call_output_label_);
          *arcp = Arc(ilabel, olabel, arc.weight, nt_nextstate);
        } else {
          return false;
        }
      } else {  // Not a non-terminal after all: expands local FST, just as above.
        // This should never happen in ActiveReplaceFst!
        LOG(WARNING) << "Encountered non-terminal not found in ActiveReplaceFst fst_array_!";
        const auto nextstate =
            flags & kArcNextStateValue
                ? state_table_->FindState(
                      StateTuple(tuple.prefix_id, tuple.fst_id, arc.nextstate))
                : kNoStateId;
        *arcp = Arc(arc.ilabel, arc.olabel, arc.weight, nextstate);
      }
    }
    return true;
  }

  // Returns the arc iterator flags supported by this FST.
  uint32 ArcIteratorFlags() const {
    uint32 flags = kArcValueFlags;
    if (!always_cache_) flags |= kArcNoCache;
    return flags;
  }

  StateTable *GetStateTable() const { return state_table_.get(); }

  const Fst<Arc> *GetFst(Label fst_id) const {
    return fst_array_[fst_id].get();
  }

  Label GetFstId(Label nonterminal) const {
    const auto it = nonterminal_hash_.find(nonterminal);
    if (it == nonterminal_hash_.end()) {
      FSTERROR() << "ActiveReplaceFstImpl::GetFstId: Nonterminal not found: "
                 << nonterminal;
    }
    return it->second;
  }

  // Returns true if label type on call arc results in epsilon input label.
  bool EpsilonOnCallInput() { return EpsilonOnInput(call_label_type_); }

 private:
  // The unique index into stack prefix table.
  PrefixId GetPrefixId(const StackPrefix &prefix) {
    return state_table_->FindPrefixId(prefix);
  }

  // The prefix ID after a stack pop.
  PrefixId PopPrefix(StackPrefix prefix) {
    prefix.Pop();
    return GetPrefixId(prefix);
  }

  // The prefix ID after a stack push.
  PrefixId PushPrefix(StackPrefix prefix, Label fst_id, StateId nextstate) {
    prefix.Push(fst_id, nextstate);
    return GetPrefixId(prefix);
  }

  // Runtime options
  ReplaceLabelType call_label_type_;    // How to label call arc.
  ReplaceLabelType return_label_type_;  // How to label return arc.
  // constexpr ReplaceLabelType call_label_type_ = REPLACE_LABEL_OUTPUT;    // How to label call arc.
  // constexpr ReplaceLabelType return_label_type_ = REPLACE_LABEL_OUTPUT;  // How to label return arc.
  int64 call_output_label_;  // Specifies output label to put on call arc
  int64 return_label_;       // Specifies label to put on return arc.
  bool always_cache_;        // Disable optional caching of arc iterator?

  // State table.
  std::unique_ptr<StateTable> state_table_;

  // Replace components.
  std::set<Label> nonterminal_set_;
  NonTerminalHash nonterminal_hash_;  // Maps non-terminal Label to fst_id (index into fst_array_).
  std::vector<std::unique_ptr<const Fst<Arc>>> fst_array_;  // Index-0 is always nullptr.
  Label root_;  // Index into fst_array_ of root.

  // Active components.
  std::vector<bool> fst_activity_;  // Parallel to fst_array_.
};

}  // namespace internal

//
// ActiveReplaceFst supports dynamic replacement of arcs in one FST with another FST.
// This replacement is recursive. ActiveReplaceFst can be used to support a variety of
// delayed constructions such as recursive
// transition networks, union, or closure. It is constructed with an array of
// FST(s). One FST represents the root (or topology) machine. The root FST
// refers to other FSTs by recursively replacing arcs labeled as non-terminals
// with the matching non-terminal FST. Currently the ActiveReplaceFst uses the output
// symbols of the arcs to determine whether the arc is a non-terminal arc or
// not. A non-terminal can be any label that is not a non-zero terminal label in
// the output alphabet.
//
// Note that the constructor uses a vector of pairs. These correspond to the
// tuple of non-terminal Label and corresponding FST. For example to implement
// the closure operation we need 2 FSTs. The first root FST is a single
// self-loop arc on the start state.
//
// The ActiveReplaceFst class supports an optionally caching arc iterator.
//
// The ActiveReplaceFst needs to be built such that it is known to be ilabel- or
// olabel-sorted (see usage below).
//
// Observe that Matcher<Fst<A>> will use the optionally caching arc iterator
// when available (the FST is ilabel-sorted and matching on the input, or the
// FST is olabel -orted and matching on the output).  In order to obtain the
// most efficient behaviour, it is recommended to set call_label_type to
// REPLACE_LABEL_INPUT or REPLACE_LABEL_BOTH and return_label_type to
// REPLACE_LABEL_OUTPUT or REPLACE_LABEL_NEITHER. This means that the call arc
// does not have epsilon on the input side and the return arc has epsilon on the
// input side) and matching on the input side.
//
// This class attaches interface to implementation and handles reference
// counting, delegating most methods to ImplToFst.
template <class A, class T = DefaultReplaceStateTable<A>,
          class CacheStore = DefaultActiveCacheStore<A>>
class ActiveReplaceFst
    : public ImplToFst<internal::ActiveReplaceFstImpl<A, T, CacheStore>> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StateTable = T;
  using Store = CacheStore;
  using State = typename CacheStore::State;
  using Impl = internal::ActiveReplaceFstImpl<Arc, StateTable, CacheStore>;
  using CacheImpl = internal::CacheBaseImpl<State, CacheStore>;

  using ImplToFst<Impl>::Properties;

  friend class ArcIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>;
  friend class StateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>;
  friend class ActiveReplaceFstMatcher<Arc, StateTable, CacheStore>;

  ActiveReplaceFst(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_array,
             Label root)
      : ImplToFst<Impl>(std::make_shared<Impl>(
            fst_array, ActiveReplaceFstOptions<Arc, StateTable, CacheStore>(root))) {}

  ActiveReplaceFst(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_array,
             const ActiveReplaceFstOptions<Arc, StateTable, CacheStore> &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst_array, opts)) {}

  // See Fst<>::Copy() for doc.
  ActiveReplaceFst(const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst,
             bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this ActiveReplaceFst. See Fst<>::Copy() for further doc.
  ActiveReplaceFst<Arc, StateTable, CacheStore> *Copy(
      bool safe = false) const override {
    return new ActiveReplaceFst<Arc, StateTable, CacheStore>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  MatcherBase<Arc> *InitMatcher(MatchType match_type) const override {
    // Active specialization!!! Always fall back to the default matcher
    // (SortedMatcher over the generic caching ArcIterator) rather than the
    // non-caching ActiveReplaceFstMatcher. There are two independent reasons,
    // and the second one is easy to miss:
    //
    // 1. Composition and lookahead must expand states through the caching path
    //    so that every visited state is cached with its activity-volatility
    //    flag set (see Expand()), which the compose-layer invalidation
    //    (ActiveComposeFst::UpdateActivity) currently depends on.
    //
    // 2. Restoring the matcher does not pay. Removing reason 1 -- for instance
    //    by deriving volatility from graph structure rather than from cache
    //    contents -- is therefore not on its own grounds to enable it.
    //    Replace states are expanded and cached by
    //    AltSequenceComposeFilter::SetState(), which evaluates NumArcs,
    //    NumInputEpsilons and Final on the fst2 state before any matching
    //    happens, so no choice of matcher avoids the caching that reason 1
    //    describes. Measured: latency and resident set unchanged. And because
    //    InitMatchers() is O(sub-FSTs x nonterminals), enabling it alongside
    //    pre-registered rule slots costs 57 s and 10.2 GiB at startup.
    //    See kaldi-active-grammar/docs/architectural-decisions.md, 2026-08-30.
    return nullptr;
  }

  bool CyclicDependencies() const { return GetImpl()->CyclicDependencies(); }

  const StateTable &GetStateTable() const {
    return *GetImpl()->GetStateTable();
  }

  const Fst<Arc> &GetFst(Label nonterminal) const {
    return *GetImpl()->GetFst(GetImpl()->GetFstId(nonterminal));
  }

  // Active specialization!!!
  const CacheStore *GetCacheStore() { return GetMutableImpl()->GetCacheStore(); }

  // Active specialization!!! Sets which sub-FSTs (by nonterminal label) are
  // active, and garbage-collects cached states whose arcs depend on activity.
  // If incremental is false, clears the entire cache instead (baseline mode).
  void UpdateActivity(const std::set<int32>& activity_set, bool incremental = true) {
    auto* impl = GetMutableImpl();
    if (incremental) {
      impl->GetCacheStore()->GCNonterminalStates();
    } else {
      impl->GetCacheStore()->Clear();
    }
    auto& activity = impl->fst_activity_;
    activity.assign(activity.size(), false);
    // activity[impl->root_] = true;  // Root is always active. Does this matter? I don't think so.
    for (const auto nonterm_label : activity_set) {
      const auto it = impl->nonterminal_hash_.find(nonterm_label);
      if (it == impl->nonterminal_hash_.end()) {
        LOG(WARNING) << "ActiveReplaceFst::UpdateActivity: no FST for nonterminal label " << nonterm_label;
        continue;
      }
      activity[it->second] = true;
    }
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  ActiveReplaceFst &operator=(const ActiveReplaceFst &) = delete;
};

// Specialization for ActiveReplaceFst.
template <class Arc, class StateTable, class CacheStore>
class StateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>
    : public CacheStateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>> {
 public:
  explicit StateIterator(const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst)
      : CacheStateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>(
            fst, fst.GetMutableImpl()) {}
};

// Specialization for ActiveReplaceFst, implementing optional caching. It is be used
// as follows:
//
//   ActiveReplaceFst<A> replace;
//   ArcIterator<ActiveReplaceFst<A>> aiter(replace, s);
//   // Note: ArcIterator< Fst<A>> is always a caching arc iterator.
//   aiter.SetFlags(kArcNoCache, kArcNoCache);
//   // Uses the arc iterator, no arc will be cached, no state will be expanded.
//   // Arc flags can be used to decide which component of the arc need to be
//   computed.
//   aiter.SetFlags(kArcILabelValue, kArcValueFlags);
//   // Wants the ilabel for this arc.
//   aiter.Value();  // Does not compute the destination state.
//   aiter.Next();
//   aiter.SetFlags(kArcNextStateValue, kArcNextStateValue);
//   // Wants the ilabel and next state for this arc.
//   aiter.Value();  // Does compute the destination state and inserts it
//                   // in the replace state table.
//   // No additional arcs have been cached at this point.
template <class Arc, class StateTable, class CacheStore>
class ArcIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>> {
 public:
  using StateId = typename Arc::StateId;

  using StateTuple = typename StateTable::StateTuple;

  ArcIterator(const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst, StateId s)
      : fst_(fst),
        s_(s),
        pos_(0),
        offset_(0),
        flags_(kArcValueFlags),
        arcs_(nullptr),
        data_flags_(0),
        final_flags_(0) {
    cache_data_.ref_count = nullptr;
    local_data_.ref_count = nullptr;
    // If FST does not support optional caching, forces caching.
    if (!(fst_.GetImpl()->ArcIteratorFlags() & kArcNoCache) &&
        !(fst_.GetImpl()->HasArcs(s_))) {
      fst_.GetMutableImpl()->Expand(s_);
    }
    // If state is already cached, use cached arcs array.
    if (fst_.GetImpl()->HasArcs(s_)) {
      (fst_.GetImpl())
          ->internal::template CacheBaseImpl<
              typename CacheStore::State,
              CacheStore>::InitArcIterator(s_, &cache_data_);
      num_arcs_ = cache_data_.narcs;
      arcs_ = cache_data_.arcs;      // arcs_ is a pointer to the cached arcs.
      data_flags_ = kArcValueFlags;  // All the arc member values are valid.
    } else {  // Otherwise delay decision until Value() is called.
      tuple_ = fst_.GetImpl()->GetStateTable()->Tuple(s_);
      if (tuple_.fst_state == kNoStateId) {
        num_arcs_ = 0;
      } else {
        // The decision to cache or not to cache has been defered until Value()
        // or
        // SetFlags() is called. However, the arc iterator is set up now to be
        // ready for non-caching in order to keep the Value() method simple and
        // efficient.
        const auto *rfst = fst_.GetImpl()->GetFst(tuple_.fst_id);
        rfst->InitArcIterator(tuple_.fst_state, &local_data_);
        // arcs_ is a pointer to the arcs in the underlying machine.
        arcs_ = local_data_.arcs;
        // Computes the final arc (but not its destination state) if a final arc
        // is required.
        bool has_final_arc = fst_.GetMutableImpl()->ComputeFinalArc(
            tuple_, &final_arc_, kArcValueFlags & ~kArcNextStateValue);
        // Sets the arc value flags that hold for final_arc_.
        final_flags_ = kArcValueFlags & ~kArcNextStateValue;
        // Computes the number of arcs.
        num_arcs_ = local_data_.narcs;
        if (has_final_arc) ++num_arcs_;
        // Sets the offset between the underlying arc positions and the
        // positions
        // in the arc iterator.
        offset_ = num_arcs_ - local_data_.narcs;
        // Defers the decision to cache or not until Value() or SetFlags() is
        // called.
        data_flags_ = 0;
      }
    }
  }

  ~ArcIterator() {
    if (cache_data_.ref_count) --(*cache_data_.ref_count);
    if (local_data_.ref_count) --(*local_data_.ref_count);
  }

  void ExpandAndCache() const  {
    // TODO(allauzen): revisit this.
    // fst_.GetImpl()->Expand(s_, tuple_, local_data_);
    // (fst_.GetImpl())->CacheImpl<A>*>::InitArcIterator(s_,
    //                                               &cache_data_);
    //
    fst_.InitArcIterator(s_, &cache_data_);  // Expand and cache state.
    arcs_ = cache_data_.arcs;      // arcs_ is a pointer to the cached arcs.
    data_flags_ = kArcValueFlags;  // All the arc member values are valid.
    offset_ = 0;                   // No offset.
  }

  void Init() {
    if (flags_ & kArcNoCache) {  // If caching is disabled
      // arcs_ is a pointer to the arcs in the underlying machine.
      arcs_ = local_data_.arcs;
      // Sets the arcs value flags that hold for arcs_.
      data_flags_ = kArcWeightValue;
      if (!fst_.GetMutableImpl()->EpsilonOnCallInput()) {
        data_flags_ |= kArcILabelValue;
      }
      // Sets the offset between the underlying arc positions and the positions
      // in the arc iterator.
      offset_ = num_arcs_ - local_data_.narcs;
    } else {
      ExpandAndCache();
    }
  }

  bool Done() const { return pos_ >= num_arcs_; }

  const Arc &Value() const {
    // If data_flags_ is 0, non-caching was not requested.
    if (!data_flags_) {
      // TODO(allauzen): Revisit this.
      if (flags_ & kArcNoCache) {
        // Should never happen.
        FSTERROR() << "ActiveReplaceFst: Inconsistent arc iterator flags";
      }
      ExpandAndCache();
    }
    if (pos_ - offset_ >= 0) {  // The requested arc is not the final arc.
      const auto &arc = arcs_[pos_ - offset_];
      if ((data_flags_ & flags_) == (flags_ & kArcValueFlags)) {
        // If the value flags match the recquired value flags then returns the
        // arc.
        return arc;
      } else {
        // Otherwise, compute the corresponding arc on-the-fly.
        fst_.GetMutableImpl()->ComputeArc(tuple_, arc, &arc_, nullptr,
                                          flags_ & kArcValueFlags);
        return arc_;
      }
    } else {  // The requested arc is the final arc.
      if ((final_flags_ & flags_) != (flags_ & kArcValueFlags)) {
        // If the arc value flags that hold for the final arc do not match the
        // requested value flags, then
        // final_arc_ needs to be updated.
        fst_.GetMutableImpl()->ComputeFinalArc(tuple_, &final_arc_,
                                               flags_ & kArcValueFlags);
        final_flags_ = flags_ & kArcValueFlags;
      }
      return final_arc_;
    }
  }

  void Next() { ++pos_; }

  size_t Position() const { return pos_; }

  void Reset() { pos_ = 0; }

  void Seek(size_t pos) { pos_ = pos; }

  uint32 Flags() const { return flags_; }

  void SetFlags(uint32 flags, uint32 mask) {
    // Updates the flags taking into account what flags are supported
    // by the FST.
    flags_ &= ~mask;
    flags_ |= (flags & fst_.GetImpl()->ArcIteratorFlags());
    // If non-caching is not requested (and caching has not already been
    // performed), then flush data_flags_ to request caching during the next
    // call to Value().
    if (!(flags_ & kArcNoCache) && data_flags_ != kArcValueFlags) {
      if (!fst_.GetImpl()->HasArcs(s_)) data_flags_ = 0;
    }
    // If data_flags_ has been flushed but non-caching is requested before
    // calling Value(), then set up the iterator for non-caching.
    if ((flags & kArcNoCache) && (!data_flags_)) Init();
  }

 private:
  const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst_;  // Reference to the FST.
  StateId s_;                                           // State in the FST.
  mutable StateTuple tuple_;  // Tuple corresponding to state_.

  ssize_t pos_;             // Current position.
  mutable ssize_t offset_;  // Offset between position in iterator and in arcs_.
  ssize_t num_arcs_;        // Number of arcs at state_.
  uint32 flags_;            // Behavorial flags for the arc iterator
  mutable Arc arc_;         // Memory to temporarily store computed arcs.

  mutable ArcIteratorData<Arc> cache_data_;  // Arc iterator data in cache.
  mutable ArcIteratorData<Arc> local_data_;  // Arc iterator data in local FST.

  mutable const Arc *arcs_;     // Array of arcs.
  mutable uint32 data_flags_;   // Arc value flags valid for data in arcs_.
  mutable Arc final_arc_;       // Final arc (when required).
  mutable uint32 final_flags_;  // Arc value flags valid for final_arc_.

  ArcIterator(const ArcIterator &) = delete;
  ArcIterator &operator=(const ArcIterator &) = delete;
};

template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFstMatcher : public MatcherBase<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FST = ActiveReplaceFst<Arc, StateTable, CacheStore>;
  using LocalMatcher = MultiEpsMatcher<Matcher<Fst<Arc>>>;

  using StateTuple = typename StateTable::StateTuple;

  // This makes a copy of the FST.
  ActiveReplaceFstMatcher(const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst,
                    MatchType match_type)
      : owned_fst_(fst.Copy()),
        fst_(*owned_fst_),
        impl_(fst_.GetMutableImpl()),
        s_(fst::kNoStateId),
        match_type_(match_type),
        current_loop_(false),
        final_arc_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId) {
    if (match_type_ == fst::MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
    InitMatchers();
  }

  // This doesn't copy the FST.
  ActiveReplaceFstMatcher(const ActiveReplaceFst<Arc, StateTable, CacheStore> *fst,
                    MatchType match_type)
      : fst_(*fst),
        impl_(fst_.GetMutableImpl()),
        s_(fst::kNoStateId),
        match_type_(match_type),
        current_loop_(false),
        final_arc_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId) {
    if (match_type_ == fst::MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
    InitMatchers();
  }

  // This makes a copy of the FST.
  ActiveReplaceFstMatcher(
      const ActiveReplaceFstMatcher<Arc, StateTable, CacheStore> &matcher,
      bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        impl_(fst_.GetMutableImpl()),
        s_(fst::kNoStateId),
        match_type_(matcher.match_type_),
        current_loop_(false),
        final_arc_(false),
        loop_(fst::kNoLabel, 0, Weight::One(), fst::kNoStateId) {
    if (match_type_ == fst::MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
    InitMatchers();
  }

  // Creates a local matcher for each component FST in the RTN. LocalMatcher is
  // a multi-epsilon wrapper matcher. MultiEpsilonMatcher is used to match each
  // non-terminal arc, since these non-terminal
  // turn into epsilons on recursion.
  void InitMatchers() {
    const auto &fst_array = impl_->fst_array_;
    matcher_.resize(fst_array.size());
    for (Label i = 0; i < fst_array.size(); ++i) {
      if (fst_array[i]) {
        matcher_[i].reset(
            new LocalMatcher(*fst_array[i], match_type_, kMultiEpsList));
        auto it = impl_->nonterminal_set_.begin();
        for (; it != impl_->nonterminal_set_.end(); ++it) {
          matcher_[i]->AddMultiEpsLabel(*it);
        }
      }
    }
  }

  ActiveReplaceFstMatcher<Arc, StateTable, CacheStore> *Copy(
      bool safe = false) const override {
    return new ActiveReplaceFstMatcher<Arc, StateTable, CacheStore>(*this, safe);
  }

  MatchType Type(bool test) const override {
    if (match_type_ == MATCH_NONE) return match_type_;
    const auto true_prop =
        match_type_ == MATCH_INPUT ? kILabelSorted : kOLabelSorted;
    const auto false_prop =
        match_type_ == MATCH_INPUT ? kNotILabelSorted : kNotOLabelSorted;
    const auto props = fst_.Properties(true_prop | false_prop, test);
    if (props & true_prop) {
      return match_type_;
    } else if (props & false_prop) {
      return MATCH_NONE;
    } else {
      return MATCH_UNKNOWN;
    }
  }

  const Fst<Arc> &GetFst() const override { return fst_; }

  uint64 Properties(uint64 props) const override { return props; }

  // Sets the state from which our matching happens.
  void SetState(StateId s) final {
    if (s_ == s) return;
    s_ = s;
    tuple_ = impl_->GetStateTable()->Tuple(s_);
    if (tuple_.fst_state == kNoStateId) {
      done_ = true;
      return;
    }
    // Gets current matcher, used for non-epsilon matching.
    current_matcher_ = matcher_[tuple_.fst_id].get();
    current_matcher_->SetState(tuple_.fst_state);
    loop_.nextstate = s_;
    final_arc_ = false;
  }

  // Searches for label from previous set state. If label == 0, first
  // hallucinate an epsilon loop; otherwise use the underlying matcher to
  // search for the label or epsilons. Note since the ActiveReplaceFst recursion
  // on non-terminal arcs causes epsilon transitions to be created we use
  // MultiEpsilonMatcher to search for possible matches of non-terminals. If the
  // component FST
  // reaches a final state we also need to add the exiting final arc.
  // Unlike stock ReplaceFst, ActiveReplaceFst deletes arcs into nonterminals
  // that are currently inactive: ComputeArc() returns false and leaves the
  // output arc untouched. Expand() honours that by not pushing the arc, so a
  // matcher has to honour it by stepping over the underlying component arc.
  // Positions the underlying matcher on the next arc that survives ComputeArc,
  // leaving it in arc_; returns false if none does.
  bool SkipInactiveArcs() {
    while (!current_matcher_->Done()) {
      if (impl_->ComputeArc(tuple_, current_matcher_->Value(), &arc_)) return true;
      current_matcher_->Next();
    }
    return false;
  }

  bool Find(Label label) final {
    bool found = false;
    label_ = label;
    if (label_ == 0 || label_ == kNoLabel) {
      // Computes loop directly, avoiding Replace::ComputeArc.
      if (label_ == 0) {
        current_loop_ = true;
        found = true;
      }
      // Searches for matching multi-epsilons. Nonterminal arcs are reached
      // through here (InitMatchers registers every nonterminal as a
      // multi-epsilon label), so this is where inactive ones must be skipped.
      final_arc_ = impl_->ComputeFinalArc(tuple_, nullptr);
      const bool component = current_matcher_->Find(kNoLabel) && SkipInactiveArcs();
      found = component || final_arc_ || found;
    } else {
      // Searches on a sub machine directly using sub machine matcher.
      found = current_matcher_->Find(label_) && SkipInactiveArcs();
    }
    return found;
  }

  bool Done() const final {
    return !current_loop_ && !final_arc_ && current_matcher_->Done();
  }

  const Arc &Value() const final {
    if (current_loop_) return loop_;
    if (final_arc_) {
      // Separate storage: arc_ holds the component arc that SkipInactiveArcs()
      // validated, and is still needed once the final arc has been consumed.
      impl_->ComputeFinalArc(tuple_, &final_arc_value_);
      return final_arc_value_;
    }
    return arc_;
  }

  void Next() final {
    if (current_loop_) {
      current_loop_ = false;
      return;
    }
    if (final_arc_) {
      final_arc_ = false;
      return;
    }
    current_matcher_->Next();
    SkipInactiveArcs();
  }

  ssize_t Priority(StateId s) final { return fst_.NumArcs(s); }

 private:
  std::unique_ptr<const ActiveReplaceFst<Arc, StateTable, CacheStore>> owned_fst_;
  const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst_;
  internal::ActiveReplaceFstImpl<Arc, StateTable, CacheStore> *impl_;
  LocalMatcher *current_matcher_;
  std::vector<std::unique_ptr<LocalMatcher>> matcher_;
  StateId s_;             // Current state.
  Label label_;           // Current label.
  MatchType match_type_;  // Supplied by caller.
  mutable bool done_;
  mutable bool current_loop_;  // Current arc is the implicit loop.
  mutable bool final_arc_;     // Current arc for exiting recursion.
  mutable StateTuple tuple_;   // Tuple corresponding to state_.
  mutable Arc arc_;             // Current component arc, validated by SkipInactiveArcs().
  mutable Arc final_arc_value_; // Current exit-recursion arc.
  Arc loop_;

  ActiveReplaceFstMatcher &operator=(const ActiveReplaceFstMatcher &) = delete;
};

template <class Arc, class StateTable, class CacheStore>
inline void ActiveReplaceFst<Arc, StateTable, CacheStore>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base =
      new StateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>(*this);
}

using StdActiveReplaceFst = ActiveReplaceFst<StdArc>;

// Recursively replaces arcs in the root FSTs with other FSTs.
// This version writes the result of replacement to an output MutableFst.
//
// Replace supports replacement of arcs in one Fst with another FST. This
// replacement is recursive. Replace takes an array of FST(s). One FST
// represents the root (or topology) machine. The root FST refers to other FSTs
// by recursively replacing arcs labeled as non-terminals with the matching
// non-terminal FST. Currently Replace uses the output symbols of the arcs to
// determine whether the arc is a non-terminal arc or not. A non-terminal can be
// any label that is not a non-zero terminal label in the output alphabet.
//
// Note that input argument is a vector of pairs. These correspond to the tuple
// of non-terminal Label and corresponding FST.
template <class Arc>
void ActiveReplace(const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
                 &ifst_array,
             MutableFst<Arc> *ofst,
             ActiveReplaceFstOptions<Arc> opts = ActiveReplaceFstOptions<Arc>()) {
  opts.gc = true;
  opts.gc_limit = 0;  // Caches only the last state for fastest copy.
  *ofst = ActiveReplaceFst<Arc>(ifst_array, opts);
}

template <class Arc>
void ActiveReplace(const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
                 &ifst_array,
             MutableFst<Arc> *ofst, const ReplaceUtilOptions &opts) {
  ActiveReplace(ifst_array, ofst, ActiveReplaceFstOptions<Arc>(opts));
}

// For backwards compatibility.
template <class Arc>
void ActiveReplace(const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
                 &ifst_array,
             MutableFst<Arc> *ofst, typename Arc::Label root,
             bool epsilon_on_replace) {
  ActiveReplace(ifst_array, ofst, ActiveReplaceFstOptions<Arc>(root, epsilon_on_replace));
}

template <class Arc>
void ActiveReplace(const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
                 &ifst_array,
             MutableFst<Arc> *ofst, typename Arc::Label root) {
  ActiveReplace(ifst_array, ofst, ActiveReplaceFstOptions<Arc>(root));
}

}  // namespace fst

#endif  // FST_ACTIVE_REPLACE_H_
