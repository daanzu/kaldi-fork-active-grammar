// fst-export

// Copyright   2019  David Zurow

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License
// for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "fstext/fstext-lib.h"
#include "fst/script/compile.h"
#include "util/common-utils.h"

#include "utils.h"
#include "md5.h"

extern "C" {
#include "dragonfly.h"
}

using namespace fst;
using StateId = StdArc::StateId;
using Weight = StdArc::Weight;
using Label = StdArc::Label;

// FIXME: this is static and is used by all!
static std::unordered_set<Label> eps_like_ilabels;
static std::unordered_set<Label> silent_olabels;
static std::unordered_set<Label> wildcard_olabels;

bool fst__init(int32_t eps_like_ilabels_len, int32_t eps_like_ilabels_cp[], int32_t silent_olabels_len, int32_t silent_olabels_cp[], int32_t wildcard_olabels_len, int32_t wildcard_olabels_cp[]) {
    eps_like_ilabels.clear();
    for (int32_t i = 0; i < eps_like_ilabels_len; ++i)
        eps_like_ilabels.emplace(eps_like_ilabels_cp[i]);
    silent_olabels.clear();
    for (int32_t i = 0; i < silent_olabels_len; ++i)
        silent_olabels.emplace(silent_olabels_cp[i]);
    wildcard_olabels.clear();
    for (int32_t i = 0; i < wildcard_olabels_len; ++i)
        wildcard_olabels.emplace(wildcard_olabels_cp[i]);
    return true;
}

void* fst__construct() {
    auto fst = new StdVectorFst();
    if (fst->AddState() != 0) KALDI_ERR << "wrong start state";
    fst->SetStart(0);
    return fst;
}

bool fst__destruct(void* fst_vp) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    delete fst;
    return true;
}

int32_t fst__add_state(void* fst_vp, float weight, bool initial) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    auto state_id = fst->AddState();
    fst->SetFinal(state_id, weight);
    if (initial)
        fst->AddArc(0, StdArc(0, 0, Weight::One(), state_id));
    return state_id;
}

bool fst__add_arc(void* fst_vp, int32_t src_state_id, int32_t dst_state_id, int32_t ilabel, int32_t olabel, float weight) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    fst->AddArc(src_state_id, StdArc(ilabel, olabel, weight, dst_state_id));
    return true;
}

bool fst__add_arcs(void* fst_vp, int32_t num_arcs, const int32_t src_state_ids_cp[], const int32_t dst_state_ids_cp[], const int32_t ilabels_cp[], const int32_t olabels_cp[], const float weights_cp[]) {
    if (fst_vp == nullptr || num_arcs < 0)
        return false;
    if (num_arcs == 0)
        return true;
    if (src_state_ids_cp == nullptr || dst_state_ids_cp == nullptr || ilabels_cp == nullptr || olabels_cp == nullptr || weights_cp == nullptr)
        return false;

    auto fst = static_cast<StdVectorFst*>(fst_vp);
    const auto num_states = fst->NumStates();
    for (int32_t i = 0; i < num_arcs; ++i) {
        if (src_state_ids_cp[i] < 0 || src_state_ids_cp[i] >= num_states ||
                dst_state_ids_cp[i] < 0 || dst_state_ids_cp[i] >= num_states)
            return false;
    }
    for (int32_t i = 0; i < num_arcs; ++i) {
        fst->AddArc(src_state_ids_cp[i], StdArc(
            ilabels_cp[i], olabels_cp[i], weights_cp[i], dst_state_ids_cp[i]));
    }
    return true;
}

bool fst__compute_md5(void* fst_vp, char* md5_cp, const char* dependencies_seed_md5_cp) {
    // ExecutionTimer timer("fst__compute_md5", -2);
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    MD5 md5;
    md5.add(dependencies_seed_md5_cp, MD5::HashBytes * 2);  // Encoded in hex text

    ArcSort(fst, ILabelCompare<StdArc>());
    // timer.step("ilabel sorted");

    static uint32_t ffff = UINT32_MAX;
    const auto nstates = fst->NumStates();
    md5.add(&nstates, sizeof(nstates));
    for (StateIterator<StdVectorFst> siter(*fst); !siter.Done(); siter.Next()) {
        const auto state = siter.Value();
        md5.add(&state, sizeof(state));
        const auto narcs = fst->NumArcs(state);
        md5.add(&narcs, sizeof(narcs));
        for (ArcIterator<StdVectorFst> aiter(*fst, state); !aiter.Done(); aiter.Next()) {
            const auto& arc = aiter.Value();
            md5.add(&ffff, sizeof(ffff));
            md5.add(&arc.nextstate, sizeof(arc.nextstate));
            md5.add(&arc.ilabel, sizeof(arc.ilabel));
            md5.add(&arc.olabel, sizeof(arc.olabel));
            md5.add(&arc.weight, sizeof(arc.weight));
        }
    }

    auto digest = md5.getHash();
    strncpy(md5_cp, digest.c_str(), MD5::HashBytes * 2 + 1);  // Encoded in hex text

    return true;
}

bool fst__has_path(void* fst_vp) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    auto path_src_state = fst->Start();
    std::deque<StateId> state_queue = { path_src_state };
    std::unordered_set<StateId> queued_states = { path_src_state };
    while (!state_queue.empty()) {
        const auto state = state_queue.front();
        state_queue.pop_front();
        if (fst->Final(state) != Weight::Zero())
            return true;
        for (ArcIterator<StdFst> aiter(*fst, state); !aiter.Done(); aiter.Next()) {
            const auto& arc = aiter.Value();
            if (!queued_states.count(arc.nextstate)) {
                state_queue.emplace_back(arc.nextstate);
                queued_states.emplace(arc.nextstate);
            }
        }
    }
    return false;
}

bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    std::deque<StateId> state_queue = { path_src_state };
    std::unordered_set<StateId> queued_states = { path_src_state };
    while (!state_queue.empty()) {
        const auto state = state_queue.front();
        state_queue.pop_front();
        if (state == path_dst_state)
            return true;
        for (ArcIterator<StdFst> aiter(*fst, state); !aiter.Done(); aiter.Next()) {
            const auto& arc = aiter.Value();
            if (eps_like_ilabels.count(arc.ilabel) && !queued_states.count(arc.nextstate)) {
                state_queue.emplace_back(arc.nextstate);
                queued_states.emplace(arc.nextstate);
            }
        }
    }
    return false;
}

bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len) {
    // NOTE: This does not support transitioning to other FSTs via nonterminals!
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    using Path = std::vector<Label>;  // olabels
    using Entry = std::tuple<StateId, Path, size_t>;  // (state, path, target_label_index)
    std::deque<Entry> queue = { std::make_tuple(fst->Start(), Path(), 0) };

    while (!queue.empty()) {
        StateId state;
        Path path;
        size_t target_label_index;
        std::tie(state, path, target_label_index) = queue.front();
        queue.pop_front();
        auto target_label = (target_label_index < target_labels_len) ? target_labels_cp[target_label_index] : -1;

        if ((target_label == -1) && (fst->Final(state) != Weight::Zero())) {
            // Found a path match ending at a final state, so return true with olabels path.
            for (auto i = 0; i < std::min((int32_t)path.size(), *output_labels_len); ++i) {
                output_labels_cp[i] = path[i];
            }
            if (path.size() > *output_labels_len)
                KALDI_WARN << "fst__does_match: output_labels_len < " << path.size();
            *output_labels_len = path.size();
            return true;
        }

        for (ArcIterator<StdFst> aiter(*fst, state); !aiter.Done(); aiter.Next()) {
            auto arc = aiter.Value();
            if ((target_label != -1) && (arc.ilabel == target_label)) {
                // Found a label match, so follow arc and enqueue next state.
                Path next_path(path);
                next_path.emplace_back(arc.olabel);
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, next_path, target_label_index+1));
            } else if (wildcard_olabels.count(arc.ilabel)) {
                // This arc can accept anything.
                if (std::find(path.begin(), path.end(), arc.olabel) == path.end()) {
                    // We haven't already added this olabel to the path, so add it now.
                    // FIXME: Is this right? shouldn't we only check for olabel at end of path?
                    path.emplace_back(arc.olabel);
                }
                if (target_label != -1) {
                    // There is a label to match, so accept it and stay at this state and re-enqueue.
                    Path next_path(path);
                    next_path.emplace_back(target_label);
                    queue.emplace_back(std::forward_as_tuple(state, next_path, target_label_index+1));
                }
                // Follow arc without taking a label, and enqueue next state.
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, path, target_label_index));
            } else if (silent_olabels.count(arc.ilabel)) {
                // Take an epsilon transition to the next state.
                Path next_path(path);
                next_path.emplace_back(arc.olabel);
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, next_path, target_label_index));
            }
        }
    }
    return false;
}

void* fst__load_file(char* filename_cp) {
    auto fst = CastOrConvertToVectorFst(ReadFstKaldiGeneric(std::string(filename_cp)));
    return fst;
}

bool fst__write_file(void* fst_vp, char* filename_cp) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    fst->Write(std::string(filename_cp));
    return true;
}

bool fst__write_file_const(void* fst_vp, char* filename_cp) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    fst::ConstFst<StdArc> const_fst(*fst);
    const_fst.Write(std::string(filename_cp));
    return true;
}

bool fst__print(void* fst_vp, char* filename_cp) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    if (filename_cp) KALDI_WARN << "printing to file not supported";
    fst::FstPrinter<StdArc> fstprinter(*fst, nullptr, nullptr, nullptr, false, false, " ");
    fstprinter.Print(&cout, "fst__print");
    return true;
}


void* fst__compile_text(char* fst_text_cp, char* isymbols_file_cp, char* osymbols_file_cp) {
    ExecutionTimer timer("fst__compile_text:compiling");
    std::istringstream fst_text(fst_text_cp);
    auto isymbols = fst::SymbolTable::ReadText(isymbols_file_cp),
        osymbols = fst::SymbolTable::ReadText(osymbols_file_cp);
    auto fstclass = fst::script::CompileFstInternal(fst_text, "<fst__compile_text>", "vector", "standard",
        isymbols, osymbols, nullptr, false, false, false, false, false);
    delete isymbols;
    delete osymbols;
    auto fst = dynamic_cast<StdVectorFst*>(fst::Convert(*fstclass->GetFst<StdArc>(), "vector"));
    if (!fst) KALDI_ERR << "could not convert Fst to StdVectorFst";
    return fst;
}

bool utils__build_L_disambig(char* lexicon_fst_text_cp, char* isymbols_file_cp, char* osymbols_file_cp, char* wdisambig_phones_file_cp, char* wdisambig_words_file_cp, char* fst_out_file_cp) {
    auto fst = static_cast<StdVectorFst*>(fst__compile_text(lexicon_fst_text_cp, isymbols_file_cp, osymbols_file_cp));

    std::vector<int32> disambig_in;
    if (!kaldi::ReadIntegerVectorSimple(wdisambig_phones_file_cp, &disambig_in))
      KALDI_ERR << "utils__build_L_disambig: Could not read disambiguation symbols from "
                 << kaldi::PrintableRxfilename(wdisambig_phones_file_cp);
    std::vector<int32> disambig_out;
    if (!kaldi::ReadIntegerVectorSimple(wdisambig_words_file_cp, &disambig_out))
      KALDI_ERR << "utils__build_L_disambig: Could not read disambiguation symbols from "
                << kaldi::PrintableRxfilename(wdisambig_words_file_cp);
    if (disambig_in.size() != disambig_out.size())
      KALDI_ERR << "utils__build_L_disambig: mismatch in size of disambiguation symbols";
    AddSelfLoops(fst, disambig_in, disambig_out);

    ArcSort(fst, OLabelCompare<StdArc>());
    WriteFstKaldi(*fst, fst_out_file_cp);
    delete fst;
    return true;
}
