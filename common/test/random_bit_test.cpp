/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <catch2/catch.hpp>

#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "common_defs.hpp"

namespace datasketches {

// returns the given values in a loop
template<typename UInt, UInt Min, UInt Max>
class sequence_generator {
public:
  using result_type = UInt;
  static constexpr result_type min() { return Min; }
  static constexpr result_type max() { return Max; }

  explicit sequence_generator(std::vector<UInt> values): values_(std::move(values)), calls_(0) {}
  result_type operator()() { return values_[calls_++ % values_.size()]; }
  size_t calls() const { return calls_; }

private:
  std::vector<UInt> values_;
  size_t calls_;
};

struct signed_result {
  using result_type = int32_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 1; }
  result_type operator()() { return 0; }
};

struct bool_result {
  using result_type = bool;
  static constexpr result_type min() { return false; }
  static constexpr result_type max() { return true; }
  result_type operator()() { return false; }
};

struct mismatched_result {
  using result_type = uint32_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 1; }
  uint64_t operator()() { return 0; }
};

struct no_result_type {
  static constexpr uint32_t min() { return 0; }
  static constexpr uint32_t max() { return 1; }
  uint32_t operator()() { return 0; }
};

struct runtime_range {
  using result_type = uint32_t;
  static result_type min() { return 0; }
  static result_type max() { return 1; }
  result_type operator()() { return 0; }
};

struct empty_range {
  using result_type = uint32_t;
  static constexpr result_type min() { return 1; }
  static constexpr result_type max() { return 1; }
  result_type operator()() { return 1; }
};

using default_random_bit_generator = decltype(random_utils::random_bit);

static_assert(is_uniform_random_bit_generator<default_random_bit_generator>::value, "");
static_assert(is_uniform_random_bit_generator<std::mt19937>::value, "");
static_assert(is_uniform_random_bit_generator<std::mt19937_64>::value, "");
static_assert(is_uniform_random_bit_generator<std::minstd_rand>::value, "");
static_assert(is_uniform_random_bit_generator<std::random_device>::value, "");
static_assert(is_uniform_random_bit_generator<sequence_generator<uint8_t, 5, 8>>::value, "");

// integers must not be taken for generators, such as a weight passed to kll_sketch::update()
static_assert(!is_uniform_random_bit_generator<uint32_t>::value, "");
static_assert(!is_uniform_random_bit_generator<int>::value, "");
static_assert(!is_uniform_random_bit_generator<const uint64_t>::value, "");
// the range is needed to derive a fair bit, so callables without it are rejected
static_assert(!is_uniform_random_bit_generator<bool()>::value, "");
static_assert(!is_uniform_random_bit_generator<std::function<bool()>>::value, "");
static_assert(!is_uniform_random_bit_generator<no_result_type>::value, "");
static_assert(!is_uniform_random_bit_generator<runtime_range>::value, "");
static_assert(!is_uniform_random_bit_generator<empty_range>::value, "");
static_assert(!is_uniform_random_bit_generator<signed_result>::value, "");
static_assert(!is_uniform_random_bit_generator<bool_result>::value, "");
static_assert(!is_uniform_random_bit_generator<mismatched_result>::value, "");
static_assert(!is_uniform_random_bit_generator<const std::mt19937>::value, "a const generator cannot be invoked");

TEST_CASE("is uniform random bit generator: lambda", "[random_bit]") {
  auto lambda = [] { return true; };
  static_assert(!is_uniform_random_bit_generator<decltype(lambda)>::value, "");
  unused(lambda);
}

TEST_CASE("random bit from: range {0, 1} gives the output", "[random_bit]") {
  sequence_generator<uint32_t, 0, 1> generator({0, 1, 1, 0});
  REQUIRE_FALSE(random_bit_from(generator));
  REQUIRE(random_bit_from(generator));
  REQUIRE(random_bit_from(generator));
  REQUIRE_FALSE(random_bit_from(generator));
  REQUIRE(generator.calls() == 4);
}

TEST_CASE("random bit from: default generator gives its output", "[random_bit]") {
  default_random_bit_generator generator(7);
  default_random_bit_generator copy(generator);
  for (int i = 0; i < 1000; ++i) REQUIRE(random_bit_from(generator) == (copy() == 1));
}

TEST_CASE("random bit from: upper half of the range", "[random_bit]") {
  sequence_generator<uint8_t, 5, 8> small({5, 6, 7, 8});
  REQUIRE_FALSE(random_bit_from(small));
  REQUIRE_FALSE(random_bit_from(small));
  REQUIRE(random_bit_from(small));
  REQUIRE(random_bit_from(small));

  const uint64_t top_bit = UINT64_C(1) << 63;
  sequence_generator<uint64_t, 0, UINT64_MAX> full({0, top_bit - 1, top_bit, UINT64_MAX});
  REQUIRE_FALSE(random_bit_from(full));
  REQUIRE_FALSE(random_bit_from(full));
  REQUIRE(random_bit_from(full));
  REQUIRE(random_bit_from(full));
  REQUIRE(full.calls() == 4);
}

TEST_CASE("random bit from: odd number of values skips the largest", "[random_bit]") {
  sequence_generator<uint16_t, 10, 12> generator({12, 10, 12, 12, 11});
  REQUIRE_FALSE(random_bit_from(generator));
  REQUIRE(generator.calls() == 2);
  REQUIRE(random_bit_from(generator));
  REQUIRE(generator.calls() == 5);
}

template<typename Engine>
static unsigned count_ones(Engine engine, unsigned num_bits) {
  unsigned ones = 0;
  for (unsigned i = 0; i < num_bits; ++i) ones += random_bit_from(engine) ? 1 : 0;
  return ones;
}

TEST_CASE("random bit from: standard engines are fair", "[random_bit]") {
  // the standard deviation of the count is about 158, and the engines are seeded, so the result is fixed
  REQUIRE(count_ones(std::mt19937(1), 100000) == Approx(50000).margin(1000));
  REQUIRE(count_ones(std::minstd_rand(1), 100000) == Approx(50000).margin(1000));
  // an odd number of values: 0 to 2^31 - 2
  REQUIRE(count_ones(std::linear_congruential_engine<uint32_t, 48271, 1, 2147483647>(1), 100000)
      == Approx(50000).margin(1000));
}

} /* namespace datasketches */
