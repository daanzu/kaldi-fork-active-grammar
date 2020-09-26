// Utils

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

#include <chrono>
#include <string>
#include <sstream>
#include "base/kaldi-error.h"

// Adapted from wav2letter and https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c/48220627#48220627
class ExecutionTimer {
   public:
    // using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    typedef std::conditional<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>::type Clock;

   private:
    const Clock::time_point start_time_ = Clock::now();
    Clock::time_point last_time_ = Clock::now();
    int32 step_count_ = 0;
    const std::string description_;
    const int32 verbosity_;
    bool stopped_ = false;

   public:
    ExecutionTimer(std::string description, int32 verbosity, bool verbose_start = false) : description_(std::move(description)), verbosity_(verbosity) {
        if (verbose_start) KALDI_VLOG(verbosity_) << "ExecutionTimer: Started " << description_ << "...";
    }
    ExecutionTimer(std::string description, bool verbose_start, int32 verbosity = 1) : ExecutionTimer(description, verbosity, verbose_start) {}
    ExecutionTimer(std::string description) : ExecutionTimer(description, 1) {}

    ~ExecutionTimer() {
        if (!stopped_) stop();
    }

    inline void stop() {
        const auto end_time = Clock::now();
        KALDI_VLOG(verbosity_) << "ExecutionTimer: " << description_ << " completed in " << pretty_duration(start_time_, end_time);
        stopped_ = true;
    }

    inline void step(const std::string& step_description = "") {
        const auto now_time = Clock::now();
        step_count_ += 1;
        KALDI_VLOG(verbosity_) << "ExecutionTimer: " << description_ << " completed Step "
            << ((!step_description.empty()) ? step_description : ("#" + std::to_string(step_count_)))
            << " with Split " << pretty_duration(start_time_, now_time)
            << " and Lap " << pretty_duration(last_time_, now_time);
        last_time_ = now_time;
    }

   private:
    std::string pretty_duration(const std::chrono::time_point<Clock>& start, const std::chrono::time_point<Clock>& end) {
        const auto runtime = end - start;
        auto runtimeMicroSec = std::chrono::duration_cast<std::chrono::microseconds>(runtime);
        auto runtimeMiliSec = std::chrono::duration_cast<std::chrono::milliseconds>(runtime);
        auto runtimeSeconds = std::chrono::duration_cast<std::chrono::seconds>(runtime);
        std::stringstream strStream;
        if (runtimeMicroSec.count() < 1e5) {
            strStream << runtimeMicroSec.count() << " microseconds";
        } else if (runtimeMiliSec.count() < 1e5) {
            strStream << runtimeMiliSec.count() << " milliseconds";
        } else {
            strStream << runtimeSeconds.count() << " seconds";
        }
        return strStream.str();
    }

};  // ExecutionTimer
