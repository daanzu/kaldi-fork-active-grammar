
#pragma once

#include "fst/compose.h"

#include "active-cache.h"


namespace fst {

template <class A, class CacheStore = DefaultActiveCacheStore<A>>
class ActiveComposeFst : public ComposeFst<A, CacheStore> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;

  using Base = ComposeFst<A, CacheStore>;
  using Base::Base;

  Fst<A>& GetFst1Unsafe() { return const_cast<Fst<A>&>(static_cast<ImplSpecialized*>(GetMutableImpl())->GetFst1()); }
  Fst<A>& GetFst2Unsafe() { return const_cast<Fst<A>&>(static_cast<ImplSpecialized*>(GetMutableImpl())->GetFst2()); }

  void SetNonterminals(Label min, Label max) { GetMutableImpl()->GetCacheStore()->SetNonterminals(min, max); }

  void UpdateActivity() {
    auto impl = GetMutableImpl();
    auto store = impl->GetCacheStore();
    store->Clear();
    // store->GCNonterminalStates();
  }

 protected:
  using Impl = internal::ComposeFstImplBase<A, CacheStore>;
  using ImplToFst<Impl>::GetMutableImpl;

  // Assumption! See ComposeFst::CreateBase().
  // using M = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::FstMatcher;
  using Filter = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::ComposeFilter;
  using ImplSpecialized = internal::ComposeFstImpl<CacheStore, Filter, GenericComposeStateTable<Arc, typename Filter::FilterState>>;
};

}
