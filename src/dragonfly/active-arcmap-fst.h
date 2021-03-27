
#pragma once

#include "fst/arc-map.h"

#include "active-cache.h"


namespace fst {

template <class A, class B, class C>
class ActiveArcMapFst : public ArcMapFst<A, B, C> {
 public:
  using Arc1 = A;
  using Arc = B;
  using Label = typename Arc::Label;
  using Mapper = C;

  using Base = ArcMapFst<A, B, C>;
  using Base::Base;

  void SetNonterminals(Label min, Label max) { GetMutableImpl()->GetCacheStore()->SetNonterminals(min, max); }

  void UpdateActivity(const std::set<int32>& activity_set) {
    auto* impl = GetMutableImpl();
    impl->GetCacheStore()->GCNonterminalStates();
  }

 protected:
  using Impl = internal::ArcMapFstImpl<A, B, C>;
  using ImplToFst<Impl>::GetMutableImpl;
};

}
