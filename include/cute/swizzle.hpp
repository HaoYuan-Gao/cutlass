/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>                      // CUTE_HOST_DEVICE
#include <cute/container/tuple.hpp>             // cute::is_tuple
#include <cute/numeric/integral_constant.hpp>   // cute::constant
#include <cute/numeric/math.hpp>                // cute::max, cute::min
#include <cute/algorithm/tuple_algorithms.hpp>  // cute::transform_apply

namespace cute
{

// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY mask
 *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */

// 通用 XOR Swizzle 地址变换:
//
// Swizzle 的作用是在不改变 Tensor 逻辑坐标的情况下，
// 对 shared memory 的物理地址进行 bit 级重新编码。
//
// 核心变换:
//
//                  ZZZ_new = ZZZ_old xor YYY
//
// 其中:
//   - YYY: 用于参与 XOR 的地址 bit
//   - ZZZ: 被修改的地址 bit
//
// 地址 bit 示意:
//
//
//   高位地址                                 低位地址
//   |                                         |
//   v                                         v
//
//   xxxxxxxxx | YYY | xxxxx | ZZZ | xxx
//                  \             /
//                   \           /
//                    \         /
//                     XOR
//                      |
//                      v
//
//   xxxxxxxxx | YYY | xxxxx | ZZZ_new | xxx
//
//
// 经过 Swizzle 后:
//
//   - Tensor 的逻辑位置不变
//   - 元素之间的映射关系不变
//   - 但是 shared memory 中实际访问的地址发生变化
//
// 主要目的:
//   通过重新分布地址 bit，改变 shared memory bank 映射，
//   从而减少或消除 bank conflict。
//
//
// 参数说明:
//
// ---------------------------------------------------------------------
// BBits:
//
//   表示参与 XOR 运算的 bit 数量。
//   它决定 YYY 和 ZZZ 两个 bit 区域的宽度。
//
//   例如:
//
//       BBits = 3
//
//   表示:
//
//       YYY = 3 bits
//       ZZZ = 3 bits
//
//   XOR 变换:
//
//       ZZZ_new[2:0] = ZZZ_old[2:0] xor YYY[2:0]
//
//   BBits 越大，参与地址重排的 bit 越多，
//   可以影响更多 shared memory bank 映射。
//
//
// ---------------------------------------------------------------------
// MBase:
//
//   表示最低多少位地址 bit 保持不变。
//   这些 bit 不参与 Swizzle，不会被 XOR 修改。
//
//   这些低位通常表示:
//
//       1. 元素内部 byte offset
//          例如:
//             half 需要保护 bit0
//             float 需要保护 bit0~bit1
//
//       2. 数据访问对齐要求
//          例如:
//             vector load/store
//             ldmatrix
//             MMA tensor core 指令
//
//   作用:
//
//       保证 Swizzle 后的地址仍然指向正确的元素，
//       不会因为修改低位地址而访问到元素中间位置。
//
//
//   例如:
//
//       MBase = 4
//
//   表示:
//
//       address:
//
//       xxxx xxxx xxxx xxxx
//                      ^^^^
//                         |
//                         最低 4 bit 固定
//
//   Swizzle 只能修改 bit4 以上的地址。
//
//   因此可以保持:
//
//       16-byte 对齐
//
//
// ---------------------------------------------------------------------
// SShift:
//
//   表示 YYY 和 ZZZ 两组 bit 之间的距离。
//   决定哪一组 bit 作为 XOR 输入，以及 XOR 到哪一组 bit。
//
//   正数 SShift:
//
//       YYY 位于 ZZZ 的高位方向
//
//       例如:
//
//             YYY
//              |
//              | SShift
//              v
//             ZZZ
//
//       变换:
//
//             ZZZ_new = ZZZ xor YYY
//
//
//   负数 SShift:
//
//       YYY 位于 ZZZ 的低位方向
//
//       表示从低位地址 bit 获取 XOR 信息。
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0,             "MBase must be positive.");
  static_assert(num_bits >= 0,             "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk::value | zzz_msk::value);

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    // 1. offset & yyy_msk{} : 提取 yyy 对应的 bit 区域
    // 2. shiftr(offset & yyy_msk{}, msk_sft{}) : 右移 yyy 区域到 zzz 区域
    // 3. offset ^ shiftr(offset & yyy_msk{}, msk_sft{}) : 对 zzz 区域进行 XOR 变换
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }

  template <int B, int M, int S>
  CUTE_HOST_DEVICE constexpr
  auto
  operator==(Swizzle<B,M,S> const&) const
  {
    return B == BBits && M == MBase && S == SShift;
  }
};

//
// make_swizzle<0b1000, 0b0100>()         ->  Swizzle<1,2,1>
// make_swizzle<0b11000000, 0b00000110>() ->  Swizzle<2,1,5>
// 根据 yyy_msk{} 和 zzz_msk{} 计算 BBits, MBase, SShift
// 
template <uint32_t Y, uint32_t Z>
CUTE_HOST_DEVICE constexpr
auto
make_swizzle()
{
  constexpr uint32_t BZ = popcount(Y);                    // Number of swizzle bits
  constexpr uint32_t BY = popcount(Z);                    // Number of swizzle bits
  static_assert(BZ == BY, "Number of bits in Y and Z don't match");
  constexpr uint32_t TZ_Y = countr_zero(Y);               // Number of trailing zeros in Y
  constexpr uint32_t TZ_Z = countr_zero(Z);               // Number of trailing zeros in Z
  constexpr uint32_t M = cute::min(TZ_Y, TZ_Z) % 32;
  constexpr  int32_t S = int32_t(TZ_Y) - int32_t(TZ_Z);   // Difference in trailing zeros
  static_assert((Y | Z) == Swizzle<BZ,M,S>::swizzle_code, "Something went wrong.");
  return Swizzle<BZ,M,S>{};
}

template <int B0, int M0, int S0,
          int B1, int M1, int S1>
CUTE_HOST_DEVICE constexpr
auto
composition(Swizzle<B0,M0,S0>, Swizzle<B1,M1,S1>)
{
  static_assert(S0 == S1, "Can only merge swizzles of the same shift.");
  constexpr uint32_t Y = Swizzle<B0,M0,S0>::yyy_msk::value ^ Swizzle<B1,M1,S1>::yyy_msk::value;
  constexpr uint32_t Z = Swizzle<B0,M0,S0>::zzz_msk::value ^ Swizzle<B1,M1,S1>::zzz_msk::value;
  return make_swizzle<Y,Z>();

  //return ComposedFn<Swizzle<B0,M0,S0>, Swizzle<B1,M1,S1>>{};
}

//
// Utility for slicing and swizzle "offsets"
//

// For swizzle functions, it is often needed to keep track of which bits are
//   consumed and which bits are free. Furthermore, it is useful to know whether
// each of these bits is known statically or dynamically.

// MixedBits is an 32-bit unsigned integer class where some bits are known statically
//   and some bits are known dynamically. These sets of bits are disjoint and it is
//   known statically which bits are known dynamically.

// MixedBits can only be manipulated through bitwise operations

// Abstract value:  StaticInt | (dynamic_int_ & StaticFlags)

/**
MixedBits

  32-bit value
  |
  +-------- or --------+
  |                |
StaticInt        dynamic_int_
(compile time)   (runtime)


StaticFlags:
0 -> static bit
1 -> dynamic bit

cutlass 属于模板编程，所以 StaticInt 和 StaticFlags 都是编译时常量。
*/
template <uint32_t StaticInt,
          uint32_t StaticFlags>    // 0: static, 1: dynamic
struct MixedBits
{
  // Representation invariants
  static_assert(StaticFlags != 0, "Should be at least one dynamic bit in MixedBits.");
  static_assert((StaticInt & StaticFlags) == 0, "No static/dynamic overlap allowed in MixedBits.");

  uint32_t dynamic_int_;
  // assert((dynamic_int_ & ~StaticFlags) == 0);

  CUTE_HOST_DEVICE constexpr operator uint32_t() const noexcept { return StaticInt | dynamic_int_; }
};

// Return a value representing (C<s>{} | (d & C<f>)) potentially using MixedBits to track s and f.
// This maker does allow ((s & f) != 0) and enforces the MixedBits invariant before creation.

/**
s：静态 bit
f：动态 bit mask
d：动态值
          make_mixed_bits
                |
                |
      ---------------------
      |                   |
  全静态               有运行时bit

return C<x>       return MixedBits<s,f>

*/
template <auto s, class DynamicType, auto f>
CUTE_HOST_DEVICE constexpr
auto
make_mixed_bits(C<s>, DynamicType const& d, C<f>)
{
  static_assert(is_integral<DynamicType>::value);
  constexpr uint32_t new_f = uint32_t(f) & ~uint32_t(s);        // StaticBits take precedence, M<0,f>{d} | C<s>{}
  if constexpr (new_f == 0 || is_static<DynamicType>::value) {
    return C<s>{} | (d & C<new_f>{});                           // Just return a static int
  } else {
    return MixedBits<s, new_f>{uint32_t(d) & new_f};            // MixedBits
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Operators
//

// Equality
template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator==(MixedBits<S0,F0> const& m, C<S1>)
{
  return (S0 == (uint32_t(S1) & ~F0)) && (m.dynamic_int_ == (uint32_t(S1) & F0));
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator==(C<S1> s, MixedBits<S0,F0> const& m)
{
  return m == s;
}

/*
 * Boolean Algebra Laws (布尔代数公式)
 *
 * Symbols:
 *   | : OR  (逻辑或)
 *   & : AND (逻辑与)
 *   ~ : NOT (逻辑非)
 *
 *
 * 1. Identity Laws (恒等律)
 *
 *   A | 0 = A
 *   A & 1 = A
 *
 *
 * 2. Null / Dominance Laws (支配律)
 *
 *   A | 1 = 1
 *   A & 0 = 0
 *
 *
 * 3. Idempotent Laws (幂等律)
 *
 *   A | A = A
 *   A & A = A
 *
 *
 * 4. Commutative Laws (交换律)
 *
 *   A | B = B | A
 *   A & B = B & A
 *
 *
 * 5. Associative Laws (结合律)
 *
 *   (A | B) | C = A | (B | C)
 *   (A & B) & C = A & (B & C)
 *
 *
 * 6. Distributive Laws (分配律)
 *
 *   AND distributes over OR:
 *
 *       A & (B | C) = (A & B) | (A & C)
 *
 *
 *   OR distributes over AND:
 *
 *       A | (B & C) = (A | B) & (A | C)
 *
 *
 * 7. Absorption Laws (吸收律)
 *
 *   A | (A & B) = A
 *
 *   A & (A | B) = A
 *
 *
 * 8. Complement Laws (互补律)
 *
 *   A | ~A = 1
 *
 *   A & ~A = 0
 *
 *
 * 9. De Morgan's Laws (德摩根定律)
 *
 *   ~(A | B) = ~A & ~B
 *
 *   ~(A & B) = ~A | ~B
 *
 */

// Bitwise AND
template <uint32_t S0, uint32_t F0,
          uint32_t S1, uint32_t F1>
CUTE_HOST_DEVICE constexpr
auto
operator&(MixedBits<S0,F0> const& m0, MixedBits<S1,F1> const& m1)
{
  // 0X0  -> static 0
  // 001  -> dynamic，目前是 0
  // 011  -> dynamic，目前是 1
  // 1X0  -> static 1

  // 推导规则表(truth table)：参考上面的布尔代数公式
  // 用来证明/说明 MixedBits & MixedBits 这个运算应该如何实现
  //  
  // MixedBits AND expansion:
  // (S0 | D0) & (S1 | D1) = (S0 & S1) | (S0 & D1) | (D0 & S1) | (D0 & D1)
  // 
  // Truth table for (S0,D0,F0) & (S1,D1,F1) -> (S,D,F)
  //   S0D0F0  | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0      | 0X0 | 0X0 | 0X0 | 0X0 |
  //  001      | 0X0 | 001 | 001 | 001 |
  //  011      | 0X0 | 001 | 011 | 011 |
  //  1X0      | 0X0 | 001 | 011 | 1X0 |

  return make_mixed_bits(C<S0 & S1>{},
                         //(S0 | m0.dynamic_int_) & (S1 | m1.dynamic_int_),
                         ((S1 & F0) & m0.dynamic_int_) | ((S0 & F1) & m1.dynamic_int_) | (m0.dynamic_int_ & m1.dynamic_int_),
                         C<(S1 & F0) | (S0 & F1) | (F0 & F1)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator&(MixedBits<S0,F0> const& m, C<S1>)
{
  return make_mixed_bits(C<S0 & uint32_t(S1)>{},
                         m.dynamic_int_,
                         C<F0 & uint32_t(S1)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator&(C<S1> s, MixedBits<S0,F0> const& m)
{
  return m & s;
}

// Bitwise OR
template <uint32_t S0, uint32_t F0,
          uint32_t S1, uint32_t F1>
CUTE_HOST_DEVICE constexpr
auto
operator|(MixedBits<S0,F0> const& m0, MixedBits<S1,F1> const& m1)
{
  // MixedBits OR expansion:
  // (S0 | D0) | (S1 | D1) = (S0 | S1) | (S0 | D1) | (D0 | S1) | (D0 | D1)
  // 其中，D0 | D1 被 dynamic bits 吸收
  // 
  // Truth table for (S0,D0,F0) | (S1,D1,F1) -> (S,D,F)
  //   S0D0F0 | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0     | 0X0 | 001 | 011 | 1X0 |
  //  001     | 001 | 001 | 011 | 1X0 |
  //  011     | 011 | 011 | 011 | 1X0 |
  //  1X0     | 1X0 | 1X0 | 1X0 | 1X0 |

  return make_mixed_bits(C<S0 | S1>{},
                         ((~S1 & F0) & m0.dynamic_int_) | ((~S0 & F1) & m1.dynamic_int_),
                         C<(~S0 & F1) | (~S1 & F0)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator|(MixedBits<S0,F0> const& m, C<S1>)
{
  return make_mixed_bits(C<S0 |  uint32_t(S1)>{},
                         m.dynamic_int_,
                         C<F0 & ~uint32_t(S1)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator|(C<S1> s, MixedBits<S0,F0> const& m)
{
  return m | s;
}

// Bitwise XOR
/*
 * MixedBits 异或运算:
 *     (S0 | D0) ^ (S1 | D1)
 *
 * 静态部分:
 *
 *     S = (~S0 & S1 & ~F0) | (S0 & ~S1 & ~F1)
 *
 *     只有两个输入都是静态，并且值不同的时候，结果才能确定为 static bit:
 *
 *       0 ^ 1 = 1
 *       1 ^ 0 = 1
 *
 *     其它情况依赖 dynamic bit。
 *
 * 动态 mask:
 *
 *     F = F0 | F1
 *
 *     XOR 运算会保留动态属性:
 *       D ^ 0 = D
 *       D ^ 1 = ~D
 *
 *     因此任意一侧为 dynamic，
 *     结果仍然是 dynamic。
 *
 * 动态值:
 *     D = (S0 | D0) ^ (S1 | D1)
 *
 */
template <uint32_t S0, uint32_t F0,
          uint32_t S1, uint32_t F1>
CUTE_HOST_DEVICE constexpr
auto
operator^(MixedBits<S0,F0> const& m0, MixedBits<S1,F1> const& m1)
{
  // Truth table for (S0,D0,F0) ^ (S1,D1,F1) -> (S,D,F)
  //   S0D0F0 | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0     | 0X0 | 001 | 011 | 1X0 |
  //  001     | 001 | 001 | 011 | 011 |
  //  011     | 011 | 011 | 001 | 001 |
  //  1X0     | 1X0 | 011 | 001 | 0X0 |

  return make_mixed_bits(C<(~S0 & S1 & ~F0) | (S0 & ~S1 & ~F1)>{},
                         (S0 | m0.dynamic_int_) ^ (S1 | m1.dynamic_int_),
                         C<F0 | F1>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator^(MixedBits<S0,F0> const& m, C<S1>)
{
  return make_mixed_bits(C<(~S0 & uint32_t(S1) & ~F0) | (S0 & ~uint32_t(S1))>{},
                         (S0 | m.dynamic_int_) ^ uint32_t(S1),
                         C<F0>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator^(C<S1> s, MixedBits<S0,F0> const& m)
{
  return m ^ s;
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator<<(MixedBits<S0,F0> const& m, C<S1>)
{
  return make_mixed_bits(C<(S0 << S1)>{},
                         m.dynamic_int_ << S1,
                         C<(F0 << S1)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
operator>>(MixedBits<S0,F0> const& m, C<S1>)
{
  return make_mixed_bits(C<(S0 >> S1)>{},
                         m.dynamic_int_ >> S1,
                         C<(F0 >> S1)>{});
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
shiftl(MixedBits<S0,F0> const& m, C<S1> s)
{
  if constexpr (S1 >= 0) {
    return m << s;
  } else {
    return m >> -s;
  }
}

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
shiftr(MixedBits<S0,F0> const& m, C<S1> s)
{
  if constexpr (S1 >= 0) {
    return m >> s;
  } else {
    return m << -s;
  }
}

//
// Upcast and Downcast
//

template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
safe_div(MixedBits<S0,F0> const& m, C<S1> s)
{
  static_assert(has_single_bit(uint32_t(S1)), "Only divide MixedBits by powers of two.");
  return make_mixed_bits(safe_div(C<S0>{}, s),
                         safe_div(m.dynamic_int_, s),
                         safe_div(C<F0>{}, s));
}

template <uint32_t N, uint32_t S0, uint32_t F0>
CUTE_HOST_DEVICE constexpr
auto
upcast(MixedBits<S0,F0> const& m)
{
  static_assert(has_single_bit(N), "Only divide MixedBits by powers of two.");
  return safe_div(m, C<N>{});
}

template <uint32_t N, class T, __CUTE_REQUIRES(cute::is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
upcast(T const& m)
{
  return safe_div(m, C<N>{});
}

template <uint32_t N, uint32_t S0, uint32_t F0>
CUTE_HOST_DEVICE constexpr
auto
downcast(MixedBits<S0,F0> const& m)
{
  static_assert(has_single_bit(N), "Only scale MixedBits by powers of two.");
  return make_mixed_bits(C<S0 * N>{},
                         m.dynamic_int_ * N,
                         C<F0 * N>{});
}

template <uint32_t N, class T, __CUTE_REQUIRES(cute::is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
downcast(T const& m)
{
  return m * C<N>{};
}

template <uint32_t S0, uint32_t F0>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(MixedBits<S0,F0> const&)
{
  return C<uint32_t(1) << countr_zero(S0 | F0)>{};
}

template <auto v>
CUTE_HOST_DEVICE constexpr
C<v>
max_alignment(C<v> const& c)
{
  return c;
}

//
// Convert a Pow2Layout+Coord to a MixedBits
//

template <class Shape, class Stride, class Coord>
CUTE_HOST_DEVICE constexpr
auto
to_mixed_bits(Shape const& shape, Stride const& stride, Coord const& coord)
{
  if constexpr (is_tuple<Shape>::value && is_tuple<Stride>::value && is_tuple<Coord>::value) {
    static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched ranks");
    static_assert(tuple_size<Shape>::value == tuple_size<Coord >::value, "Mismatched ranks");
    // 对于 Pow2Layout，每个维度的 offset 占据不同的 bits
    // 例如 (8, 16):(16, 1) ----- dimension0: xxxx000000, dimension1: 0000xxxx
    // 所以 xor 操作就是合并 bit pattern, 即 offset = (c0​∗s0​) ^ (c1​∗s1​)..., 等价于 offset = (c0​∗s0​) + (c1​∗s1​)...,
    return transform_apply(shape, stride, coord, [](auto const& s, auto const& d, auto const& c) { return to_mixed_bits(s,d,c); },
                                                 [](auto const&... a) { return (a ^ ...); });
  } else if constexpr (is_integral<Shape>::value && is_integral<Stride>::value && is_integral<Coord>::value) {
    static_assert(decltype(shape*stride)::value == 0 || has_single_bit(decltype(shape*stride)::value), "Requires pow2 shape*stride.");
    /*
     * static bits = 0
     * dynamic value = coord * stride
     * dynamic mask = (shape - 1) * stride, 其中，shape - 1 的意思是：0 <= coord < shape
     */
    return make_mixed_bits(Int<0>{}, coord * stride, (shape - Int<1>{}) * stride);
  } else {
    static_assert(is_integral<Shape>::value && is_integral<Stride>::value && is_integral<Coord>::value, "Either Shape, Stride, and Coord must be all tuples, or they must be all integral (in the sense of cute::is_integral).");
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Layout, class Coord>
CUTE_HOST_DEVICE constexpr
auto
to_mixed_bits(Layout const& layout, Coord const& coord)
{
  return to_mixed_bits(layout.shape(), layout.stride(), idx2crd(coord, layout.shape()));
}

//
// Display utilities
//

template <int B, int M, int S>
CUTE_HOST_DEVICE void print(Swizzle<B,M,S> const&)
{
  printf("Sw<%d,%d,%d>", B, M, S);
}

template <uint32_t S, uint32_t F>
CUTE_HOST_DEVICE void print(MixedBits<S,F> const& m)
{
  printf("M_%u|(%u&%u)=%u", S, m.dynamic_int_, F, uint32_t(m));
}

#if !defined(__CUDACC_RTC__)
template <int B, int M, int S>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Swizzle<B,M,S> const&)
{
  return os << "Sw<" << B << "," << M << "," << S << ">";
}

template <uint32_t S, class D, uint32_t F>
CUTE_HOST std::ostream& operator<<(std::ostream& os, MixedBits<S,F> const& m)
{
  return os << "M_" << S << "|(" << m.dynamic_int_ << "&" << F << ")=" << uint32_t(m);
}
#endif // !defined(__CUDACC_RTC__)

//
// Helper Function
//
template <class T, class = void>                      // Default No-Swizzle
struct get_swizzle { using type = Swizzle<0,4,3>; };

template <class T>
using get_swizzle_t = typename get_swizzle<T>::type;

} // end namespace cute
