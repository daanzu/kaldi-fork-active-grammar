
#pragma once

#include "fst/arc-map.h"

#include "active-cache.h"


namespace fst {

template <class A, class B, class C, class CacheStore>
class ActiveArcMapFst;

namespace internal {

// Implementation of delayed ArcMapFst.
template <class A, class B, class C, class CacheStore = DefaultActiveCacheStore<B>>
class ActiveArcMapFstImpl : public CacheBaseImpl<CacheState<B>, CacheStore> {
 public:
  using Arc = B;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<B>::SetType;
  using FstImpl<B>::SetProperties;
  using FstImpl<B>::SetInputSymbols;
  using FstImpl<B>::SetOutputSymbols;

  using CacheBase = CacheBaseImpl<CacheState<B>, CacheStore>;
  using CacheBase::HasArcs;
  using CacheBase::HasFinal;
  using CacheBase::HasStart;
  using CacheBase::PushArc;
  using CacheBase::SetArcs;
  using CacheBase::SetFinal;
  using CacheBase::SetStart;

  friend class StateIterator<ActiveArcMapFst<A, B, C, CacheStore>>;

  ActiveArcMapFstImpl(const Fst<A> &fst, const C &mapper,
                const ArcMapFstOptions &opts)
      : CacheBase(opts),
        fst_(fst.Copy()),
        mapper_(new C(mapper)),
        own_mapper_(true),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ActiveArcMapFstImpl(const Fst<A> &fst, C *mapper, const ArcMapFstOptions &opts)
      : CacheBase(opts),
        fst_(fst.Copy()),
        mapper_(mapper),
        own_mapper_(false),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ActiveArcMapFstImpl(const ActiveArcMapFstImpl<A, B, C> &impl)
      : CacheBase(impl),
        fst_(impl.fst_->Copy(true)),
        mapper_(new C(*impl.mapper_)),
        own_mapper_(true),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ~ActiveArcMapFstImpl() override {
    if (own_mapper_) delete mapper_;
  }

  StateId Start() {
    if (!HasStart()) SetStart(FindOState(fst_->Start()));
    return CacheBase::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      switch (final_action_) {
        case MAP_NO_SUPERFINAL:
        default: {
          const auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
            FSTERROR() << "ArcMapFst: Non-zero arc labels for superfinal arc";
            SetProperties(kError, kError);
          }
          SetFinal(s, final_arc.weight);
          break;
        }
        case MAP_ALLOW_SUPERFINAL: {
          if (s == superfinal_) {
            SetFinal(s, Weight::One());
          } else {
            const auto final_arc =
                (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
            if (final_arc.ilabel == 0 && final_arc.olabel == 0) {
              SetFinal(s, final_arc.weight);
            } else {
              SetFinal(s, Weight::Zero());
            }
          }
          break;
        }
        case MAP_REQUIRE_SUPERFINAL: {
          SetFinal(s, s == superfinal_ ? Weight::One() : Weight::Zero());
          break;
        }
      }
    }
    return CacheBase::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheBase::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheBase::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheBase::NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) && (fst_->Properties(kError, false) ||
                            (mapper_->Properties(0) & kError))) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  void InitArcIterator(StateId s, ArcIteratorData<B> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheBase::InitArcIterator(s, data);
  }

  void Expand(StateId s) {
    // Add exiting arcs.
    if (s == superfinal_) {
      SetArcs(s);
      return;
    }
    for (ArcIterator<Fst<A>> aiter(*fst_, FindIState(s)); !aiter.Done();
         aiter.Next()) {
      auto aarc = aiter.Value();
      aarc.nextstate = FindOState(aarc.nextstate);
      PushArc(s, (*mapper_)(aarc));
    }

    // Check for superfinal arcs.
    if (!HasFinal(s) || Final(s) == Weight::Zero()) {
      switch (final_action_) {
        case MAP_NO_SUPERFINAL:
        default:
          break;
        case MAP_ALLOW_SUPERFINAL: {
          auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
            if (superfinal_ == kNoStateId) superfinal_ = nstates_++;
            final_arc.nextstate = superfinal_;
            PushArc(s, std::move(final_arc));
          }
          break;
        }
        case MAP_REQUIRE_SUPERFINAL: {
          const auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0 ||
              final_arc.weight != B::Weight::Zero()) {
            PushArc(s, B(final_arc.ilabel, final_arc.olabel, final_arc.weight,
                         superfinal_));
          }
          break;
        }
      }
    }
    SetArcs(s);
  }

  // Active specialization!!!
  Fst<A> *GetFstUnsafe() { return const_cast<Fst<A>*>(fst_.get()); }  // Unsafe?

 private:
  void Init() {
    SetType("map");
    if (mapper_->InputSymbolsAction() == MAP_COPY_SYMBOLS) {
      SetInputSymbols(fst_->InputSymbols());
    } else if (mapper_->InputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
      SetInputSymbols(nullptr);
    }
    if (mapper_->OutputSymbolsAction() == MAP_COPY_SYMBOLS) {
      SetOutputSymbols(fst_->OutputSymbols());
    } else if (mapper_->OutputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
      SetOutputSymbols(nullptr);
    }
    if (fst_->Start() == kNoStateId) {
      final_action_ = MAP_NO_SUPERFINAL;
      SetProperties(kNullProperties);
    } else {
      final_action_ = mapper_->FinalAction();
      uint64 props = fst_->Properties(kCopyProperties, false);
      SetProperties(mapper_->Properties(props));
      if (final_action_ == MAP_REQUIRE_SUPERFINAL) superfinal_ = 0;
    }
  }

  // Maps from output state to input state.
  StateId FindIState(StateId s) {
    if (superfinal_ == kNoStateId || s < superfinal_) {
      return s;
    } else {
      return s - 1;
    }
  }

  // Maps from input state to output state.
  StateId FindOState(StateId is) {
    auto os = is;
    if (!(superfinal_ == kNoStateId || is < superfinal_)) ++os;
    if (os >= nstates_) nstates_ = os + 1;
    return os;
  }

  std::unique_ptr<const Fst<A>> fst_;
  C *mapper_;
  const bool own_mapper_;
  MapFinalAction final_action_;
  StateId superfinal_;
  StateId nstates_;
};

}  // namespace internal

// Maps an arc type A to an arc type B using Mapper function object
// C. This version is a delayed FST.
template <class A, class B, class C, class CacheStore = DefaultActiveCacheStore<B>>
class ActiveArcMapFst : public ImplToFst<internal::ActiveArcMapFstImpl<A, B, C, CacheStore>> {
 public:
  using Arc = B;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = CacheStore;
  using State = typename Store::State;
  using Impl = internal::ActiveArcMapFstImpl<A, B, C, CacheStore>;

  friend class ArcIterator<ActiveArcMapFst<A, B, C, CacheStore>>;
  friend class StateIterator<ActiveArcMapFst<A, B, C, CacheStore>>;

  ActiveArcMapFst(const Fst<A> &fst, const C &mapper, const ArcMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  ActiveArcMapFst(const Fst<A> &fst, C *mapper, const ArcMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  ActiveArcMapFst(const Fst<A> &fst, const C &mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, ArcMapFstOptions())) {}

  ActiveArcMapFst(const Fst<A> &fst, C *mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, ArcMapFstOptions())) {}

  // See Fst<>::Copy() for doc.
  ActiveArcMapFst(const ActiveArcMapFst<A, B, C> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this ActiveArcMapFst. See Fst<>::Copy() for further doc.
  ActiveArcMapFst<A, B, C> *Copy(bool safe = false) const override {
    return new ActiveArcMapFst<A, B, C>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<B> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<B> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  // Active specialization!!!
  using Label = typename Arc::Label;

  // Active specialization!!!
  Fst<A> *GetFstUnsafe() { return GetMutableImpl()->GetFstUnsafe(); }

  // Active specialization!!!
  void SetNonterminals(Label min, Label max) { GetMutableImpl()->GetCacheStore()->SetNonterminals(min, max); }

  // Active specialization!!!
  void UpdateActivity() {
    auto impl = GetMutableImpl();
    auto store = impl->GetCacheStore();
    store->Clear();
    // store->GCNonterminalStates();
  }

 protected:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

 private:
  ActiveArcMapFst &operator=(const ActiveArcMapFst &) = delete;
};

// Specialization for ActiveArcMapFst.
//
// This may be derived from.
template <class A, class B, class C, class CacheStore>
class StateIterator<ActiveArcMapFst<A, B, C, CacheStore>> : public StateIteratorBase<B> {
 public:
  using StateId = typename B::StateId;

  explicit StateIterator(const ActiveArcMapFst<A, B, C, CacheStore> &fst)
      : impl_(fst.GetImpl()),
        siter_(*impl_->fst_),
        s_(0),
        superfinal_(impl_->final_action_ == MAP_REQUIRE_SUPERFINAL) {
    CheckSuperfinal();
  }

  bool Done() const final { return siter_.Done() && !superfinal_; }

  StateId Value() const final { return s_; }

  void Next() final {
    ++s_;
    if (!siter_.Done()) {
      siter_.Next();
      CheckSuperfinal();
    } else if (superfinal_) {
      superfinal_ = false;
    }
  }

  void Reset() final {
    s_ = 0;
    siter_.Reset();
    superfinal_ = impl_->final_action_ == MAP_REQUIRE_SUPERFINAL;
    CheckSuperfinal();
  }

 private:
  void CheckSuperfinal() {
    if (impl_->final_action_ != MAP_ALLOW_SUPERFINAL || superfinal_) return;
    if (!siter_.Done()) {
      const auto final_arc =
          (*impl_->mapper_)(A(0, 0, impl_->fst_->Final(s_), kNoStateId));
      if (final_arc.ilabel != 0 || final_arc.olabel != 0) superfinal_ = true;
    }
  }

  const internal::ActiveArcMapFstImpl<A, B, C, CacheStore> *impl_;
  StateIterator<Fst<A>> siter_;
  StateId s_;
  bool superfinal_;  // True if there is a superfinal state and not done.
};

// Specialization for ActiveArcMapFst.
template <class A, class B, class C, class CacheStore>
class ArcIterator<ActiveArcMapFst<A, B, C, CacheStore>>
    : public CacheArcIterator<ActiveArcMapFst<A, B, C, CacheStore>> {
 public:
  using StateId = typename A::StateId;

  ArcIterator(const ActiveArcMapFst<A, B, C, CacheStore> &fst, StateId s)
      : CacheArcIterator<ActiveArcMapFst<A, B, C, CacheStore>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class A, class B, class C, class CacheStore>
inline void ActiveArcMapFst<A, B, C, CacheStore>::InitStateIterator(
    StateIteratorData<B> *data) const {
  data->base = new StateIterator<ActiveArcMapFst<A, B, C, CacheStore>>(*this);
}

}
