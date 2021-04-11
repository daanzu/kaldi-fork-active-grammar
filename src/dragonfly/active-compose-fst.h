
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

  void SetNonterminals(Label min, Label max) { GetMutableImpl()->GetCacheStore()->SetNonterminals(min, max); }

  void UpdateActivity() {
    auto* impl = GetMutableImpl();
    impl->GetCacheStore()->GCNonterminalStates();
  }

 protected:
  using Impl = internal::ComposeFstImplBase<A, CacheStore>;
  using ImplToFst<Impl>::GetMutableImpl;
};

}
