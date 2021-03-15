
#pragma once

#include "fstext/fstext-lib.h"

namespace fst {

constexpr uint32 kCacheActiveChecked = 0x0010;  // Has been checked for departure.
constexpr uint32 kCacheActiveDeparture = 0x0020;  // Is a departure state.

// This class implements
template <class CacheStore>
class ActiveCacheStore {
 public:
  using State = typename CacheStore::State;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;
  using Label = typename State::Label;

  // Required constructors/assignment operators.
  explicit ActiveCacheStore(const CacheOptions &opts)
      : store_(opts),
        nonterminal_min_(0),
        nonterminal_max_(0) {
    VLOG(1) << "ActiveCacheStore Ctor: object = " << this;
  }

  // Returns 0 if state is not stored.
  const State *GetState(StateId s) const { return store_.GetState(s); }

  // Creates state if state is not stored
  State *GetMutableState(StateId s) {
    auto state = store_.GetMutableState(s);
    if (!(state->Flags() & kCacheActiveChecked)) {
      state->SetFlags(kCacheActiveChecked, kCacheActiveChecked);
      auto narcs = state->NumArcs();
      auto arcs = state->Arcs();
      for (auto arc = arcs; arc < (arcs + narcs); ++arc) {
        if (nonterminal_min_ <= arc->olabel && arc->olabel <= nonterminal_max_) {
          state->SetFlags(kCacheActiveDeparture, kCacheActiveDeparture);
          break;
        }
      }
    }
    return state;
  }

  // Similar to State::AddArc() but updates cache store book-keeping.
  void AddArc(State *state, const Arc &arc) { store_.AddArc(state, arc); }

  // Similar to State::SetArcs() but updates internal cache size; call only
  // once.
  void SetArcs(State *state) { store_.SetArcs(state); }

  // Deletes all arcs
  void DeleteArcs(State *state) { store_.DeleteArcs(state); }

  // Deletes some arcs
  void DeleteArcs(State *state, size_t n) { store_.DeleteArcs(state, n); }

  // Deletes all cached states
  void Clear() {
    store_.Clear();
  }

  StateId CountStates() const { return store_.CountStates(); }

  // Iterates over cached states (in an arbitrary order); only needed if GC is
  // enabled.
  bool Done() const { return store_.Done(); }

  StateId Value() const { return store_.Value(); }

  void Next() { store_.Next(); }

  void Reset() { store_.Reset(); }

  // Deletes current state and advances to next.
  void Delete() {
    store_.Delete();
  }

  // Removes from the cache store (not referenced-counted and not the current)
  // states that have not been accessed since the last GC until at most
  // cache_fraction * cache_limit_ bytes are cached. If that fails to free
  // enough, attempts to uncaching recently visited states as well. If still
  // unable to free enough memory, then widens cache_limit_.
  void UpdateActivity(const State *current) {
    VLOG(1) << "ActiveCacheStore: Enter GC: object = " << "(" << this << ")\n";
    store_.Reset();
    while (!store_.Done()) {
      auto *state = store_.GetMutableState(store_.Value());
      if (state != current && state->RefCount() == 0 && (state->Flags() & kCacheActiveDeparture)) {
        store_.Delete();
      } else {
        store_.Next();
      }
    }
    VLOG(1) << "ActiveCacheStore: Exit GC: object = " << "(" << this << ")\n";
  }

  void SetNonterminals(Label min, Label max) {
    nonterminal_min_ = min; nonterminal_max_ = max;
  }

 private:
  CacheStore store_;       // Underlying store.
  Label nonterminal_min_;
  Label nonterminal_max_;
};

template <class Arc>
class DefaultActiveCacheStore
    : public ActiveCacheStore<GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>>> {
 public:
  explicit DefaultActiveCacheStore(const CacheOptions &opts)
      : ActiveCacheStore<GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>>>(opts) {
  }
};

template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFstMatcher;

template <class A, class T /* = DefaultReplaceStateTable<A> */,
          class CacheStore /* = DefaultCacheStore<A> */>
class ActiveReplaceFst
    :
    public ImplToFst<internal::ReplaceFstImpl<A, T, CacheStore>>
    // public ReplaceFst<A, T, CacheStore>
     {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StateTable = T;
  using Store = CacheStore;
  using State = typename CacheStore::State;
  using Impl = internal::ReplaceFstImpl<Arc, StateTable, CacheStore>;
  using CacheImpl = internal::CacheBaseImpl<State, CacheStore>;

  using ImplToFst<Impl>::Properties;

  friend class ArcIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>;
  friend class StateIterator<ActiveReplaceFst<Arc, StateTable, CacheStore>>;
  friend class ActiveReplaceFstMatcher<Arc, StateTable, CacheStore>;

  ActiveReplaceFst(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_array,
             Label root)
      : ImplToFst<Impl>(std::make_shared<Impl>(
            fst_array, ReplaceFstOptions<Arc, StateTable, CacheStore>(root))) {}

  ActiveReplaceFst(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_array,
             const ReplaceFstOptions<Arc, StateTable, CacheStore> &opts)
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
    if ((GetImpl()->ArcIteratorFlags() & kArcNoCache) &&
        ((match_type == MATCH_INPUT && Properties(kILabelSorted, false)) ||
         (match_type == MATCH_OUTPUT && Properties(kOLabelSorted, false)))) {
      return new ActiveReplaceFstMatcher<Arc, StateTable, CacheStore>
          (this, match_type);
    } else {
      VLOG(2) << "Not using replace matcher";
      return nullptr;
    }
  }

  bool CyclicDependencies() const { return GetImpl()->CyclicDependencies(); }

  const StateTable &GetStateTable() const {
    return *GetImpl()->GetStateTable();
  }

  const Fst<Arc> &GetFst(Label nonterminal) const {
    return *GetImpl()->GetFst(GetImpl()->GetFstId(nonterminal));
  }

  void Test() {
    auto y = GetMutableImpl()->GetCacheStore()->SetNonterminals(99,999);
    auto x = GetMutableImpl()->GetCacheStore()->UpdateActivity();
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  ActiveReplaceFst &operator=(const ActiveReplaceFst &) = delete;
};

template <class Arc, class StateTable, class CacheStore>
class ActiveReplaceFstMatcher : public MatcherBase<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FST = ReplaceFst<Arc, StateTable, CacheStore>;
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
  // search for the label or epsilons. Note since the ReplaceFst recursion
  // on non-terminal arcs causes epsilon transitions to be created we use
  // MultiEpsilonMatcher to search for possible matches of non-terminals. If the
  // component FST
  // reaches a final state we also need to add the exiting final arc.
  bool Find(Label label) final {
    bool found = false;
    label_ = label;
    if (label_ == 0 || label_ == kNoLabel) {
      // Computes loop directly, avoiding Replace::ComputeArc.
      if (label_ == 0) {
        current_loop_ = true;
        found = true;
      }
      // Searches for matching multi-epsilons.
      final_arc_ = impl_->ComputeFinalArc(tuple_, nullptr);
      found = current_matcher_->Find(kNoLabel) || final_arc_ || found;
    } else {
      // Searches on a sub machine directly using sub machine matcher.
      found = current_matcher_->Find(label_);
    }
    return found;
  }

  bool Done() const final {
    return !current_loop_ && !final_arc_ && current_matcher_->Done();
  }

  const Arc &Value() const final {
    if (current_loop_) return loop_;
    if (final_arc_) {
      impl_->ComputeFinalArc(tuple_, &arc_);
      return arc_;
    }
    const auto &component_arc = current_matcher_->Value();
    impl_->ComputeArc(tuple_, component_arc, &arc_);
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
  }

  ssize_t Priority(StateId s) final { return fst_.NumArcs(s); }

 private:
  std::unique_ptr<const ActiveReplaceFst<Arc, StateTable, CacheStore>> owned_fst_;
  const ActiveReplaceFst<Arc, StateTable, CacheStore> &fst_;
  internal::ReplaceFstImpl<Arc, StateTable, CacheStore> *impl_;
  LocalMatcher *current_matcher_;
  std::vector<std::unique_ptr<LocalMatcher>> matcher_;
  StateId s_;             // Current state.
  Label label_;           // Current label.
  MatchType match_type_;  // Supplied by caller.
  mutable bool done_;
  mutable bool current_loop_;  // Current arc is the implicit loop.
  mutable bool final_arc_;     // Current arc for exiting recursion.
  mutable StateTuple tuple_;   // Tuple corresponding to state_.
  mutable Arc arc_;
  Arc loop_;

  ActiveReplaceFstMatcher &operator=(const ActiveReplaceFstMatcher &) = delete;
};

}
