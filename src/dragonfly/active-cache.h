
#pragma once

#include "fstext/fstext-lib.h"

namespace fst {

constexpr uint32 kCacheActiveChecked = 0x0010;  // Has been checked for volatility.
constexpr uint32 kCacheActiveVolatile = 0x0020;  // Is a active-volatile state.

constexpr uint32 kActiveCacheSpecialLabel = 2000000;  // Special label indicating a volatile state.

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
        nonterminal_min_(-1),
        nonterminal_max_(-1) {
    VLOG(1) << "ActiveCacheStore Ctor: object = " << this;
  }

  // Returns 0 if state is not stored.
  const State *GetState(StateId s) const { return store_.GetState(s); }

  // Creates state if state is not stored
  State *GetMutableState(StateId s) { return store_.GetMutableState(s); }

  void SetStateActiveVolatility(StateId s, bool activeVolatile) { SetStateActiveVolatility(GetMutableState(s), activeVolatile); }

  void SetStateActiveVolatility(State *state, bool activeVolatile) {
    state->SetFlags(kCacheActiveChecked, kCacheActiveChecked);
    state->SetFlags((activeVolatile ? kCacheActiveVolatile : 0), kCacheActiveVolatile);
  }

  // Similar to State::AddArc() but updates cache store book-keeping.
  // void AddArc(State *state, const Arc &arc) { store_.AddArc(state, arc); }

  // Similar to State::SetArcs() but updates internal cache size; call only
  // once.
  void SetArcs(State *state) { store_.SetArcs(state); }

  // Deletes all arcs
  void DeleteArcs(State *state) { store_.DeleteArcs(state); }

  // Deletes some arcs
  void DeleteArcs(State *state, size_t n) { store_.DeleteArcs(state, n); }

  // Deletes all cached states
  void Clear() {
    VLOG(1) << "ActiveCacheStore: Clearing: object = " << "(" << this << ")"
      ", count_states = " << CountStates() << ", cache_size = " << store_.CacheSize() << ", cache_limit = " << store_.CacheLimit();
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

  bool IsVolatile(StateId s) const {
    const auto *state = GetState(s);
    return state && IsVolatile(state);
  }

  inline bool IsVolatile(const State *state) const { return (state->Flags() & kCacheActiveVolatile); }

  inline bool IsVolatile1(const State *state) const {
    for (auto *arc = state->Arcs(), *end_arcs = arc + state->NumArcs(); arc < end_arcs; ++arc) {
      if (nonterminal_min_ <= arc->olabel && arc->olabel <= nonterminal_max_) {
        return true;
      }
    }
    return false;
  }

  inline bool IsVolatile2(const State *state) const {
    for (auto *arc = state->Arcs(), *end_arcs = arc + state->NumArcs(); arc < end_arcs; ++arc) {
      if (arc->olabel == kActiveCacheSpecialLabel) {
        return true;
      }
    }
    return false;
  }

  void GCNonterminalStates() {
    VLOG(1) << "ActiveCacheStore: Enter GCNonterminalStates: object = " << "(" << this << ")"
      ", count_states = " << CountStates() << ", cache_size = " << store_.CacheSize() << ", cache_limit = " << store_.CacheLimit();
    uint32 num_deleted = 0;
    store_.Reset();
    while (!store_.Done()) {
      auto *state = store_.GetMutableState(store_.Value());

      if (false && !(state->Flags() & kCacheActiveChecked)) {
        KALDI_WARN << "State not checked for nonterminal! " << state;
        // if (!(state->Flags() & kCacheArcs)) {
        //   KALDI_WARN << "State does not have cached arcs! " << state;
        // } else {
        //   if (IsVolatile(state)) {
        //     state->SetFlags(kCacheActiveVolatile, kCacheActiveVolatile);
        //   }
        //   state->SetFlags(kCacheActiveChecked, kCacheActiveChecked);
        // }
      }

      if (state->Flags() & kCacheActiveVolatile) {
        if (state->RefCount() == 0) {
          // FIXME: we could be smarter about this and only delete states where the activity changed.
          store_.Delete();
          num_deleted++;
        } else {
          KALDI_WARN << "Nonterminal state not free to GC! " << state;
          store_.Delete();
          num_deleted++;
          // store_.Next();
        }
      } else {
        store_.Next();
      }
    }
    VLOG(1) << "ActiveCacheStore: Exit GCNonterminalStates: object = " << "(" << this << "), num_deleted = " << num_deleted;
  }

  // Must be called BEFORE creating any states with outgoing nonterminal arcs! Assumes nonterminal labels are contiguous!
  void SetNonterminals(Label min, Label max) {
    nonterminal_min_ = min; nonterminal_max_ = max;
  }

 private:
  CacheStore store_;       // Underlying store.
  Label nonterminal_min_;  // Defines the range of labels where all are non-terminals.
  Label nonterminal_max_;  // Defines the range of labels where all are non-terminals.
  // std::unordered_set<StateId> volatile_states_;
};

template <class Arc>
class DefaultActiveCacheStore
    : public ActiveCacheStore<GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>>> {
 public:
  explicit DefaultActiveCacheStore(const CacheOptions &opts)
      : ActiveCacheStore<GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>>>(opts) {
  }
};

}
