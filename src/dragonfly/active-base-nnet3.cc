// NNet3 Active Base

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

#include "feat/wave-reader.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice-lexicon.h"
#include "nnet3/nnet-utils.h"
#include "fst/script/compile.h"

#include "active-base-nnet3.h"
#include "nlohmann_json.hpp"

namespace dragonfly {

using namespace kaldi;
using namespace fst;

ActiveBaseNNet3OnlineModelWrapper::ActiveBaseNNet3OnlineModelWrapper(ActiveBaseNNet3OnlineModelConfig::Ptr config, int32 verbosity)
    : BaseNNet3OnlineModelWrapper(config, verbosity), config_(config) {
}

ActiveBaseNNet3OnlineModelWrapper::~ActiveBaseNNet3OnlineModelWrapper() {
}

} // namespace dragonfly


extern "C" {
#include "dragonfly.h"
}

using namespace dragonfly;
