//===-- unittests/Common/EnumClassTest.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Common/enum-class.h"

namespace {

ENUM_CLASS(Determinism, Det, Semidet, Multi, Nondet, CcMulti, CcNondet,
    Erroneous, Failure)

TEST(EnumClassTest, Introspection) {
  EXPECT_EQ(Determinism_enumSize, 8u);
  EXPECT_EQ(EnumToString(Determinism::Det), "Det");
  EXPECT_EQ(EnumToString(Determinism::Failure), "Failure");
  int count{0};
  ForEachDeterminism([&](Determinism) { ++count; });
  EXPECT_EQ(count, 8);
}

} // namespace
