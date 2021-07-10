
#pragma once

#include "fst/compose.h"

#include "active-cache.h"


namespace fst {

template <typename S, typename FS>
class ActiveComposeStateTuple : public DefaultComposeStateTuple<S, FS> {
 public:
  using ActiveComposeStateTuple::ActiveComposeStateTuple;

  bool checked = false;
};

// A HashStateTable over composition tuples.
template <typename Arc, typename FilterState,
          typename T =
              ActiveComposeStateTuple<typename Arc::StateId, FilterState>,
          typename H = ComposeHash<T>,
          typename StateTable =
              CompactHashBiTable<typename T::StateId, T, H>>
class ActiveComposeStateTable : public StateTable {
 public:
  using StateTuple = T;
  using StateId = typename Arc::StateId;

  using CompactHashBiTable<StateId, StateTuple, H>::FindId;
  using CompactHashBiTable<StateId, StateTuple, H>::FindEntry;
  using CompactHashBiTable<StateId, StateTuple, H>::Size;

  ActiveComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {}

  ActiveComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                           size_t table_size)
      : StateTable(table_size) {}

  constexpr bool Error() const { return false; }

  // Creates state in the table if necessary.
  StateId FindState(const StateTuple &tuple) {
    // auto size = Size();
    auto id = FindId(tuple);
    // if (Size() > size) { new_states_.push_back(id); }
    // if (id > highest_state_) { highest_state_ = id; }
    return id;
  }

  // State must already exist in the table.
  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }

  // std::vector<StateId> new_states_;
  // StateId highest_state_ = kNoStateId;

 private:
  ActiveComposeStateTable &operator=(const ActiveComposeStateTable &table) =
      delete;
};

// template<class Arc, class FilterState>
// using ActiveComposeStateTable = GenericComposeStateTable<Arc, FilterState, ActiveComposeStateTuple<typename Arc::StateId, FilterState>>;

template<class Arc,
  class M = Matcher<Fst<Arc>>,
  class Filter = SequenceComposeFilter<M>,
  class FilterState = typename Filter::FilterState>
using ActiveComposeFstOptions = ComposeFstOptions<Arc, M, Filter, ActiveComposeStateTable<Arc, FilterState>>;

template <class A, class CacheStore = DefaultActiveCacheStore<A>>
class ActiveComposeFst : public ComposeFst<A, CacheStore> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  using Base = ComposeFst<A, CacheStore>;
  using Base::Base;

  Fst<A>& GetFst1Unsafe() { return const_cast<Fst<A>&>(GetMutableImplSpecialized()->GetFst1()); }
  Fst<A>& GetFst2Unsafe() { return const_cast<Fst<A>&>(GetMutableImplSpecialized()->GetFst2()); }

  void SetNonterminals(Label min, Label max) {
    GetMutableImpl()->GetCacheStore()->SetNonterminals(min, max);
    next_highest_state_to_process_ = 0;
  }

  void UpdateActivity() {
    auto impl = GetMutableImpl();
    auto store = impl->GetCacheStore();

    if (true) {
      store->Clear();
    } else if (true) {
      auto* impl = GetMutableImplSpecialized();
      auto* state_table = impl->GetStateTable();
      auto* active_replace_fst = static_cast<ActiveReplaceFst<StdArc>*>(&GetFst2Unsafe());

      auto* cache_store = active_replace_fst->GetCacheStore();
      for (auto id = next_highest_state_to_process_; id < state_table->Size(); ++id) {
        const auto& tuple = state_table->Tuple(id);
        const auto s2 = tuple.StateId2();
        if (s2 >= 0 && cache_store->IsVolatile(s2)) { store->SetStateActiveVolatility(id, true); }
      }
      next_highest_state_to_process_ = state_table->Size();
      // for (const auto id : state_table->new_states_) {
      // }
      // state_table->new_states_.clear();

      store->GCNonterminalStates();
    }
  }

 protected:
  using Impl = internal::ComposeFstImplBase<A, CacheStore>;
  using ImplToFst<Impl>::GetMutableImpl;

  // Assumption on MATCH_OUTPUT! See ComposeFst::CreateBase().
  // using M = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::FstMatcher;
  using Filter = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::ComposeFilter;
  // using ImplSpecialized = internal::ComposeFstImpl<CacheStore, Filter, GenericComposeStateTable<Arc, typename Filter::FilterState>>;
  using ImplSpecialized = internal::ComposeFstImpl<CacheStore, Filter, ActiveComposeStateTable<Arc, typename Filter::FilterState>>;

  ImplSpecialized* GetMutableImplSpecialized() { return static_cast<ImplSpecialized*>(GetMutableImpl()); }

  StateId next_highest_state_to_process_ = 0;
};

}
