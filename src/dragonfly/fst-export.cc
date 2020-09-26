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

#pragma once

#include "fstext/fstext-lib.h"

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

bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    std::deque<StateId> state_queue = { path_src_state };
    std::unordered_set<StateId> queued = { path_src_state };
    while (!state_queue.empty()) {
        auto state = state_queue.front();
        state_queue.pop_front();
        if (state == path_dst_state)
            return true;
        for (ArcIterator<StdFst> aiter(*fst, state); !aiter.Done(); aiter.Next()) {
            auto arc = aiter.Value();
            if (eps_like_ilabels.count(arc.ilabel) && !queued.count(arc.nextstate)) {
                state_queue.emplace_back(arc.nextstate);
                queued.emplace(arc.nextstate);
            }
        }
    }
    return false;
}

bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len) {
    auto fst = static_cast<StdVectorFst*>(fst_vp);
    using Path = std::vector<Label>;
    using Entry = std::tuple<StateId, Path, size_t>;
    std::deque<Entry> queue = { std::make_tuple(fst->Start(), Path(), 0) };

    while (!queue.empty()) {
        StateId state;
        Path path;
        size_t target_label_index;
        std::tie(state, path, target_label_index) = queue.front();
        queue.pop_front();

        auto target_label = (target_label_index < target_labels_len) ? target_labels_cp[target_label_index] : -1;
        if ((target_label == -1) && (fst->Final(state) != Weight::Zero())) {
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
                Path next_path(path);
                next_path.emplace_back(arc.olabel);
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, next_path, target_label_index+1));
            } else if (wildcard_olabels.count(arc.ilabel)) {
                if (std::find(path.begin(), path.end(), arc.olabel) == path.end())
                    path.emplace_back(arc.olabel);
                if (target_label != -1) {
                    Path next_path(path);
                    next_path.emplace_back(target_label);
                    queue.emplace_back(std::forward_as_tuple(state, next_path, target_label_index+1));
                }
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, path, target_label_index));
            } else if (silent_olabels.count(arc.ilabel)) {
                Path next_path(path);
                next_path.emplace_back(arc.olabel);
                queue.emplace_back(std::forward_as_tuple(arc.nextstate, next_path, target_label_index));
            }
        }
    }
    return false;
}
