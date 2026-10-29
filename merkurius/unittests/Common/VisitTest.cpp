//===-- unittests/Common/VisitTest.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Common/visit.h"
#include "gtest/gtest.h"
#include "merkurius/Common/idioms.h"
#include <string>
#include <variant>

namespace common = merkurius::common;

namespace {

TEST(VisitTest, Visitors) {
  using V = std::variant<int, double, std::string>;
  auto kind{[](const V &v) {
    return common::visit(common::visitors{
                             [](int) { return 0; },
                             [](double) { return 1; },
                             [](const std::string &) { return 2; },
                         },
        v);
  }};
  EXPECT_EQ(kind(V{1}), 0);
  EXPECT_EQ(kind(V{1.0}), 1);
  EXPECT_EQ(kind(V{std::string{"a"}}), 2);
}

// More than eight alternatives exercises the binary search in log2visit.
TEST(VisitTest, ManyAlternatives) {
  using V = std::variant<char, signed char, unsigned char, short,
      unsigned short, int, unsigned, long, unsigned long, long long>;
  auto size{[](const V &v) {
    return common::visit([](auto x) { return sizeof x; }, v);
  }};
  EXPECT_EQ(size(V{'a'}), sizeof(char));
  EXPECT_EQ(size(V{short{1}}), sizeof(short));
  EXPECT_EQ(size(V{1L}), sizeof(long));
  EXPECT_EQ(size(V{1LL}), sizeof(long long));
}

} // namespace
