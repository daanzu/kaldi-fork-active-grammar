// decoder/active-grammar-fst-test.cc

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.

#include "decoder/active-grammar-fst.h"

namespace fst {

class ActiveGrammarFstTest {
 public:
  static void TestRejectsZeroStateIfst();
  static void TestNestedCallAndReturnActivity();

 private:
  struct ExpandedStateSnapshot {
    bool active;
    int32 dest_ifst_index;
    int32 dest_fst_instance;
    std::vector<StdArc> arcs;
  };

  static int32 EncodeSymbol(int32 nonterm_phones_offset,
                            int32 nonterminal,
                            int32 left_context_phone) {
    return kNontermBigNumber +
        nonterminal * GetEncodingMultiple(nonterm_phones_offset) +
        left_context_phone;
  }

  static ActiveGrammarFstArc TakeOnlyArc(const ActiveGrammarFst &fst,
                                         ActiveGrammarFst::StateId state) {
    ArcIterator<ActiveGrammarFst> iter(fst, state);
    KALDI_ASSERT(!iter.Done());
    ActiveGrammarFstArc arc = iter.Value();
    iter.Next();
    KALDI_ASSERT(iter.Done());
    return arc;
  }

  static ActiveGrammarFst::ExpandedState *FindExpandedBoundary(
      ActiveGrammarFst *fst, int32 instance_id, int32 nonterminal) {
    KALDI_ASSERT(instance_id >= 0 &&
                 static_cast<size_t>(instance_id) < fst->instances_.size());
    ActiveGrammarFst::ExpandedState *found = NULL;
    const ActiveGrammarFst::FstInstance &instance =
        fst->instances_[instance_id];
    for (const auto &item : instance.expanded_states) {
      ActiveGrammarFst::ExpandedState *expanded = item.second;
      if (expanded->nonterminal == nonterminal) {
        KALDI_ASSERT(found == NULL);
        found = expanded;
      }
    }
    KALDI_ASSERT(found != NULL);
    return found;
  }

  static ExpandedStateSnapshot Snapshot(
      const ActiveGrammarFst::ExpandedState &expanded) {
    ExpandedStateSnapshot ans;
    ans.active = expanded.active;
    ans.dest_ifst_index = expanded.dest_ifst_index;
    ans.dest_fst_instance = expanded.dest_fst_instance;
    ans.arcs = expanded.arcs;
    return ans;
  }

  static void AssertReturnUnchanged(
      const ActiveGrammarFst::ExpandedState &expanded,
      int32 nonterm_end,
      const ExpandedStateSnapshot &before) {
    KALDI_ASSERT(expanded.nonterminal == nonterm_end);
    KALDI_ASSERT(expanded.active);
    KALDI_ASSERT(expanded.active == before.active);
    KALDI_ASSERT(expanded.dest_ifst_index == before.dest_ifst_index);
    KALDI_ASSERT(expanded.dest_fst_instance == before.dest_fst_instance);
    KALDI_ASSERT(expanded.arcs.size() == before.arcs.size());
    for (size_t i = 0; i < expanded.arcs.size(); ++i) {
      const StdArc &actual = expanded.arcs[i], &expected = before.arcs[i];
      KALDI_ASSERT(actual.ilabel == expected.ilabel);
      KALDI_ASSERT(actual.olabel == expected.olabel);
      KALDI_ASSERT(actual.weight == expected.weight);
      KALDI_ASSERT(actual.nextstate == expected.nextstate);
    }
  }
};

void ActiveGrammarFstTest::TestRejectsZeroStateIfst() {
  const int32 nonterm_phones_offset = 100;
  const int32 nonterminal =
      nonterm_phones_offset + kNontermUserDefined;
  const TropicalWeight one = TropicalWeight::One();

  VectorFst<StdArc> top;
  StdArc::StateId top_state = top.AddState();
  top.SetStart(top_state);
  top.SetFinal(top_state, one);

  VectorFst<StdArc> empty_ifst;
  ConstFst<StdArc> top_const(top), empty_ifst_const(empty_ifst);
  std::vector<std::pair<int32, const ConstFst<StdArc> *> > ifsts;
  ifsts.push_back(std::make_pair(nonterminal, &empty_ifst_const));

  bool rejected = false;
  try {
    ActiveGrammarFst grammar(nonterm_phones_offset, top_const, ifsts);
  } catch (const kaldi::KaldiFatalError &error) {
    rejected = true;
    KALDI_ASSERT(std::string(error.KaldiMessage()).find(
        "zero-state rule FSTs are not supported") != std::string::npos);
  }
  KALDI_ASSERT(rejected);
}

void ActiveGrammarFstTest::TestNestedCallAndReturnActivity() {
  const int32 nonterm_phones_offset = 100;
  const int32 bos = nonterm_phones_offset + kNontermBos;
  const int32 nonterm_begin = nonterm_phones_offset + kNontermBegin;
  const int32 nonterm_end = nonterm_phones_offset + kNontermEnd;
  const int32 nonterm_reenter = nonterm_phones_offset + kNontermReenter;
  const int32 parent_nonterminal =
      nonterm_phones_offset + kNontermUserDefined;
  const int32 child_nonterminal = parent_nonterminal + 1;
  const int32 phone = 1;
  const TropicalWeight one = TropicalWeight::One();

  // top: call parent, then accept the parent's reentry context.
  VectorFst<StdArc> top;
  StdArc::StateId top_start = top.AddState(),
      top_reenter = top.AddState(), top_final = top.AddState();
  top.SetStart(top_start);
  top.SetFinal(top_final, one);
  top.AddArc(top_start,
             StdArc(EncodeSymbol(nonterm_phones_offset,
                                 parent_nonterminal, bos),
                    0, one, top_reenter));
  top.AddArc(top_reenter,
             StdArc(EncodeSymbol(nonterm_phones_offset,
                                 nonterm_reenter, phone),
                    0, one, top_final));

  // parent: enter, call child, accept its reentry context, then return.
  VectorFst<StdArc> parent;
  StdArc::StateId parent_start = parent.AddState(),
      parent_call = parent.AddState(),
      parent_reenter = parent.AddState(),
      parent_end = parent.AddState(), parent_final = parent.AddState();
  parent.SetStart(parent_start);
  parent.SetFinal(parent_final, one);
  parent.AddArc(parent_start,
                StdArc(EncodeSymbol(nonterm_phones_offset,
                                    nonterm_begin, bos),
                       0, one, parent_call));
  parent.AddArc(parent_call,
                StdArc(EncodeSymbol(nonterm_phones_offset,
                                    child_nonterminal, bos),
                       0, one, parent_reenter));
  parent.AddArc(parent_reenter,
                StdArc(EncodeSymbol(nonterm_phones_offset,
                                    nonterm_reenter, phone),
                       0, one, parent_end));
  parent.AddArc(parent_end,
                StdArc(EncodeSymbol(nonterm_phones_offset,
                                    nonterm_end, phone),
                       0, one, parent_final));

  // child: enter, consume one ordinary phone, then return to parent.
  VectorFst<StdArc> child;
  StdArc::StateId child_start = child.AddState(),
      child_body = child.AddState(), child_end = child.AddState(),
      child_final = child.AddState();
  child.SetStart(child_start);
  child.SetFinal(child_final, one);
  child.AddArc(child_start,
               StdArc(EncodeSymbol(nonterm_phones_offset,
                                   nonterm_begin, bos),
                      0, one, child_body));
  child.AddArc(child_body, StdArc(phone, phone, one, child_end));
  child.AddArc(child_end,
               StdArc(EncodeSymbol(nonterm_phones_offset,
                                   nonterm_end, phone),
                      0, one, child_final));

  PrepareForActiveGrammarFst(nonterm_phones_offset, &top);
  PrepareForActiveGrammarFst(nonterm_phones_offset, &parent);
  PrepareForActiveGrammarFst(nonterm_phones_offset, &child);

  ConstFst<StdArc> top_const(top), parent_const(parent), child_const(child);
  std::vector<std::pair<int32, const ConstFst<StdArc> *> > ifsts;
  ifsts.push_back(std::make_pair(parent_nonterminal, &parent_const));
  ifsts.push_back(std::make_pair(child_nonterminal, &child_const));
  ActiveGrammarFst grammar(nonterm_phones_offset, top_const, ifsts);

  std::vector<bool> activity(2, true);
  KALDI_ASSERT(grammar.UpdateActivity(activity));

  // Walk the only path to materialize the top call, nested call, child return,
  // and parent return in three distinct FST instances.
  ActiveGrammarFst::StateId state = grammar.Start();
  state = TakeOnlyArc(grammar, state).nextstate;  // top -> parent
  state = TakeOnlyArc(grammar, state).nextstate;  // parent -> child
  state = TakeOnlyArc(grammar, state).nextstate;  // child's ordinary phone
  state = TakeOnlyArc(grammar, state).nextstate;  // child -> parent return
  state = TakeOnlyArc(grammar, state).nextstate;  // parent -> top return
  KALDI_ASSERT(grammar.Final(state) == TropicalWeight::One());

  KALDI_ASSERT(grammar.instances_.size() == 3);
  ActiveGrammarFst::ExpandedState *top_parent_call =
      FindExpandedBoundary(&grammar, 0, parent_nonterminal);
  ActiveGrammarFst::ExpandedState *parent_child_call =
      FindExpandedBoundary(&grammar, 1, child_nonterminal);
  ActiveGrammarFst::ExpandedState *parent_return =
      FindExpandedBoundary(&grammar, 1, nonterm_end);
  ActiveGrammarFst::ExpandedState *child_return =
      FindExpandedBoundary(&grammar, 2, nonterm_end);

  KALDI_ASSERT(top_parent_call->active);
  KALDI_ASSERT(parent_child_call->active);
  KALDI_ASSERT(parent_return->active);
  KALDI_ASSERT(child_return->active);

  // dest_ifst_index alone cannot classify a boundary: the top-level call to
  // parent and the child's return to parent deliberately share index zero.
  KALDI_ASSERT(top_parent_call->dest_ifst_index == 0);
  KALDI_ASSERT(child_return->dest_ifst_index == 0);
  KALDI_ASSERT(top_parent_call->nonterminal == parent_nonterminal);
  KALDI_ASSERT(child_return->nonterminal == nonterm_end);

  const ExpandedStateSnapshot parent_return_before = Snapshot(*parent_return);
  const ExpandedStateSnapshot child_return_before = Snapshot(*child_return);

  // Parent activity controls its user-defined call boundary, but must never
  // control the already-expanded child-to-parent return boundary.
  activity[0] = false;
  KALDI_ASSERT(grammar.UpdateActivity(activity));
  KALDI_ASSERT(!top_parent_call->active);
  KALDI_ASSERT(parent_child_call->active);
  AssertReturnUnchanged(*parent_return, nonterm_end, parent_return_before);
  AssertReturnUnchanged(*child_return, nonterm_end, child_return_before);

  activity[0] = true;
  KALDI_ASSERT(grammar.UpdateActivity(activity));
  KALDI_ASSERT(top_parent_call->active);
  AssertReturnUnchanged(*parent_return, nonterm_end, parent_return_before);
  AssertReturnUnchanged(*child_return, nonterm_end, child_return_before);

  // This assertion fails with the current instance-0-only implementation.
  // A naive all-instance sweep would fix this call but fail the return checks
  // above; a structurally filtered all-instance update must satisfy both.
  activity[1] = false;
  KALDI_ASSERT(grammar.UpdateActivity(activity));
  KALDI_ASSERT(!parent_child_call->active);
  AssertReturnUnchanged(*parent_return, nonterm_end, parent_return_before);
  AssertReturnUnchanged(*child_return, nonterm_end, child_return_before);

  activity[1] = true;
  KALDI_ASSERT(grammar.UpdateActivity(activity));
  KALDI_ASSERT(parent_child_call->active);
  AssertReturnUnchanged(*parent_return, nonterm_end, parent_return_before);
  AssertReturnUnchanged(*child_return, nonterm_end, child_return_before);
}

}  // namespace fst

int main() {
  fst::ActiveGrammarFstTest::TestRejectsZeroStateIfst();
  fst::ActiveGrammarFstTest::TestNestedCallAndReturnActivity();
  KALDI_LOG << "ActiveGrammarFst tests succeeded.";
  return 0;
}
