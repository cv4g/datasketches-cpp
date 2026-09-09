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

#ifndef _COMMON_DEFS_HPP_
#define _COMMON_DEFS_HPP_

#include <cstdint>
#include <string>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <type_traits>
#include <utility>

/// DataSketches namespace
namespace datasketches {

static const uint64_t DEFAULT_SEED = 9001;

enum resize_factor { X1 = 0, X2, X4, X8 };

template<typename A> using string = std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<A>::template rebind_alloc<char>>;

// common random declarations
namespace random_utils {
  static std::random_device rd; // possibly unsafe in MinGW with GCC < 9.2
  static thread_local std::mt19937_64 rand(rd());
  static thread_local std::uniform_real_distribution<> next_double(0.0, 1.0);
  static thread_local std::uniform_int_distribution<uint64_t> next_uint64(0, UINT64_MAX);

  // thread-safe random bit
  static thread_local std::independent_bits_engine<std::mt19937, 1, uint32_t>
    random_bit(static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()
      + std::hash<std::thread::id>{}(std::this_thread::get_id())));

  inline void override_seed(uint64_t s) {
    rand.seed(s);
  }
}

template<typename... Ts> struct make_void { using type = void; };

/**
 * C++11 version of the UniformRandomBitGenerator requirements of the standard library,
 * which C++20 expresses as std::uniform_random_bit_generator plus G::result_type:
 * G::result_type is an unsigned integer type, G::min() and G::max() are constant expressions of that type
 * with G::min() < G::max(), and invoking a non-const G returns G::result_type.
 * The standard engines, such as std::mt19937_64, meet these requirements.
 */
template<typename G, typename = void>
struct is_uniform_random_bit_generator: std::false_type {};

template<typename G>
struct is_uniform_random_bit_generator<G, typename make_void<
    typename G::result_type,
    decltype(std::declval<G&>()()),
    std::integral_constant<typename G::result_type, G::min()>, // fails unless G::min() is a constant expression
    std::integral_constant<typename G::result_type, G::max()>
  >::type>: std::integral_constant<bool,
    std::is_unsigned<typename G::result_type>::value
    && !std::is_same<typename std::remove_cv<typename G::result_type>::type, bool>::value
    && std::is_same<decltype(std::declval<G&>()()), typename G::result_type>::value
    && std::is_same<decltype(G::min()), typename G::result_type>::value
    && std::is_same<decltype(G::max()), typename G::result_type>::value
    && (G::min() < G::max())
  > {};

/**
 * Returns a fair random bit from a uniform random bit generator.
 * The bit depends only on the generator output, unlike std::bernoulli_distribution,
 * so a given generator state gives the same bits with any standard library.
 * The bit tells whether the output is in the upper half of the range, since the high bits
 * of some engines are better than the low bits. If the range has an odd number of values,
 * the largest value is skipped, so the generator can be invoked more than once.
 * With a range of {0, 1}, the bit is the output.
 */
template<typename G>
bool random_bit_from(G& generator) {
  static_assert(is_uniform_random_bit_generator<G>::value, "G must be a uniform random bit generator");
  using U = typename G::result_type;
  const U span = static_cast<U>(G::max() - G::min()); // number of values - 1
  const U last = static_cast<U>(span - 1 + span % 2); // largest accepted offset: an even number of values from 0
  for (;;) {
    const U offset = static_cast<U>(generator() - G::min());
    if (offset <= last) return offset > last / 2;
  }
}

// utility function to hide unused compiler warning
// usually has no additional cost
template<typename T> void unused(T&&...) {}

// common helping functions
// TODO: find a better place for them

constexpr uint8_t log2(uint32_t n) {
  return (n > 1) ? 1 + log2(n >> 1) : 0;
}

constexpr uint8_t lg_size_from_count(uint32_t n, double load_factor) {
  return log2(n) + ((n > static_cast<uint32_t>((1 << (log2(n) + 1)) * load_factor)) ? 2 : 1);
}

// stream helpers to hide casts
template<typename T>
static inline T read(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template<typename T>
static inline void read(std::istream& is, T* ptr, size_t size_bytes) {
  is.read(reinterpret_cast<char*>(ptr), size_bytes);
}

template<typename T>
static inline void write(std::ostream& os, T value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
static inline void write(std::ostream& os, const T* ptr, size_t size_bytes) {
  os.write(reinterpret_cast<const char*>(ptr), size_bytes);
}

template<typename T>
T byteswap(T value) {
  char* ptr = static_cast<char*>(static_cast<void*>(&value));
  const int len = sizeof(T);
  for (size_t i = 0; i < len / 2; ++i) {
    std::swap(ptr[i], ptr[len - i - 1]);
  }
  return value;
}

template<typename T>
static inline T read_big_endian(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return byteswap(value);
}

// wrapper for iterators to implement operator-> returning temporary value
template<typename T>
class return_value_holder {
public:
  return_value_holder(T value): value_(value) {}
  const T* operator->() const { return std::addressof(value_); }
private:
  T value_;
};

} // namespace

#endif // _COMMON_DEFS_HPP_
