
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

}
