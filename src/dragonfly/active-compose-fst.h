
#pragma once

#include <unordered_map>

#include "fst/compose.h"
#include "fst/lookahead-filter.h"
#include "fst/lookahead-matcher.h"

#include "active-cache.h"


namespace fst {

// Lazy composition specialized for LAF decoding: fst1 is an olabel-lookahead FST
// (e.g. HCLr read as StdOLabelLookAheadFst), and fst2 is an ActiveReplaceFst whose
// arcs change when grammar activity changes.
//
// This must use the same matcher/filter stack that ComposeFst::CreateBase() would
// auto-select for a lookahead fst1 (see DefaultLookAhead<StdArc, MATCH_OUTPUT> in
// fst/lookahead-filter.h); constructing ComposeFst with explicit ComposeFstOptions
// left at their defaults would silently compose without lookahead pruning.
//
// We construct the (stock) composition state table ourselves and retain a pointer
// to it (the impl takes ownership), so that UpdateActivity() can map cached
// composition states back to their fst2 states without needing accessors on the
// impl, and without any assumptions about the impl's concrete type.
template <class A, class CacheStore = DefaultActiveCacheStore<A>>
class ActiveComposeFst : public ComposeFst<A, CacheStore> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  using Base = ComposeFst<A, CacheStore>;

  using LookAhead = DefaultLookAhead<Arc, MATCH_OUTPUT>;
  using Matcher = typename LookAhead::FstMatcher;
  using Filter = typename LookAhead::ComposeFilter;
  using FilterState = typename Filter::FilterState;
  using StateTable = GenericComposeStateTable<Arc, FilterState>;
  using Options = ComposeFstOptions<Arc, Matcher, Filter, StateTable>;

  ActiveComposeFst(const Fst<Arc>& fst1, const Fst<Arc>& fst2, const CacheOptions& cache_opts)
      : ActiveComposeFst(fst1, fst2, cache_opts, new StateTable(fst1, fst2)) {}

  // Invalidates cached composition states whose arcs may depend on grammar activity,
  // preserving the rest of the cache. Must be called BEFORE the replace-layer
  // UpdateActivity(), whose GC deletes the volatility information consulted here.
  // If incremental is false, simply clears the entire cache.
  template <class ReplaceFst>
  void UpdateActivity(ReplaceFst& replace_fst, bool incremental = true) {
    auto* store = GetMutableImpl()->GetCacheStore();
    if (!incremental) {
      store->Clear();
      return;
    }
    const auto* replace_store = replace_fst.GetCacheStore();
    std::unordered_map<StateId, bool> verdicts;  // Memoized per distinct fst2 state.
    size_t num_deleted = 0, num_kept = 0;
    store->Reset();
    while (!store->Done()) {
      const auto id = store->Value();
      const auto s2 = state_table_->Tuple(id).StateId2();
      auto it = verdicts.find(s2);
      if (it == verdicts.end())
        it = verdicts.emplace(s2, IsActivityDependent(replace_store, s2)).first;
      if (it->second) {
        store->Delete();
        ++num_deleted;
      } else {
        store->Next();
        ++num_kept;
      }
    }
    VLOG(1) << "ActiveComposeFst::UpdateActivity: num_deleted = " << num_deleted
            << ", num_kept = " << num_kept;
  }

 protected:
  using Impl = internal::ComposeFstImplBase<A, CacheStore>;
  using ImplToFst<Impl>::GetMutableImpl;

  ActiveComposeFst(const Fst<Arc>& fst1, const Fst<Arc>& fst2, const CacheOptions& cache_opts, StateTable* state_table)
      : Base(fst1, fst2, MakeOptions(cache_opts, state_table)),
        state_table_(state_table) {}

  static Options MakeOptions(const CacheOptions& cache_opts, StateTable* state_table) {
    Options opts(cache_opts);
    opts.state_table = state_table;  // Impl takes ownership.
    return opts;
  }

  // Returns true if the cached arcs of a composition state paired with fst2 state s2
  // could change when grammar activity changes: s2 itself has (possibly currently
  // dropped) nonterminal call arcs, or any of its cached successors does. The
  // successor check is required because label lookahead (and label/weight pushing)
  // inspects fst2 arcs one step beyond s2, so arcs cached at s2's composition states
  // were pruned/pushed based on the arcs of s2's successors. States absent from the
  // replace-layer cache, or cached without arcs, are conservatively treated as
  // activity-dependent (e.g. after GC eviction).
  template <class ReplaceCacheStore>
  static bool IsActivityDependent(const ReplaceCacheStore* replace_store, StateId s2) {
    if (s2 < 0) return true;
    const auto* state = replace_store->GetState(s2);
    if (!state || !(state->Flags() & kCacheArcs)) return true;
    if (replace_store->IsVolatile(state)) return true;
    for (const auto *arc = state->Arcs(), *end_arcs = arc + state->NumArcs(); arc < end_arcs; ++arc) {
      const auto* next_state = replace_store->GetState(arc->nextstate);
      if (!next_state || !(next_state->Flags() & kCacheArcs)) return true;
      if (replace_store->IsVolatile(next_state)) return true;
    }
    return false;
  }

  StateTable* state_table_;  // Owned by the impl; valid for our lifetime.

 private:
  ActiveComposeFst& operator=(const ActiveComposeFst&) = delete;
};

}
