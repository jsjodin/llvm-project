//===-- unittests/Parser/OperatorsTest.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/operators.h"
#include "gtest/gtest.h"

using namespace merkurius::parser;

namespace {

void ExpectForm(const std::optional<OperatorForm> &form, Specifier specifier,
    int priority) {
  ASSERT_TRUE(form);
  EXPECT_EQ(form->specifier, specifier);
  EXPECT_EQ(form->priority, priority);
}

TEST(OperatorsTest, Lookup) {
  // Spot checks against the manual: 22.01's priority is 1500 minus these.
  const OperatorDefinition *neck{FindOperator(":-")};
  ASSERT_TRUE(neck);
  ExpectForm(neck->prefix, Specifier::FX, 300);
  ExpectForm(neck->infix, Specifier::XFX, 300);
  const OperatorDefinition *minus{FindOperator("-")};
  ASSERT_TRUE(minus);
  ExpectForm(minus->prefix, Specifier::FX, 1300);
  ExpectForm(minus->infix, Specifier::YFX, 1000);
  ExpectForm(FindOperator(".")->infix, Specifier::YFX, 1490);
  ExpectForm(FindOperator("^")->infix, Specifier::XFY, 1401);
  ExpectForm(FindOperator("^")->prefix, Specifier::FX, 1400);
  ExpectForm(FindOperator("some")->prefix, Specifier::FXY, 550);
  ExpectForm(FindOperator("\\+")->prefix, Specifier::FY, 600);
  ExpectForm(FindOperator("mod")->infix, Specifier::XFX, 1100);
  ExpectForm(FindOperator("is")->infix, Specifier::XFX, 799);
  ExpectForm(FindOperator("--->")->infix, Specifier::XFY, 321);
  ExpectForm(FindOperator("?-")->prefix, Specifier::FX, 300);
  EXPECT_FALSE(FindOperator(",")->prefix);
  EXPECT_FALSE(FindOperator("some")->infix);
}

TEST(OperatorsTest, NotOperators) {
  EXPECT_FALSE(FindOperator("foo"));
  EXPECT_FALSE(FindOperator("|"));
  EXPECT_FALSE(FindOperator("'-'")); // callers remove quotes first
  // Additions after 22.01 (design document, "Language version").
  EXPECT_FALSE(FindOperator("<<u"));
  EXPECT_FALSE(FindOperator(">>u"));
}

TEST(OperatorsTest, ArgumentPriorities) {
  OperatorForm comma{*FindOperator(",")->infix}; // xfy
  EXPECT_EQ(comma.LeftArgumentPriority(), 501);
  EXPECT_EQ(comma.RightArgumentPriority(), 500);
  OperatorForm minus{*FindOperator("-")->infix}; // yfx
  EXPECT_EQ(minus.LeftArgumentPriority(), 1000);
  EXPECT_EQ(minus.RightArgumentPriority(), 1001);
  OperatorForm equals{*FindOperator("=")->infix}; // xfx
  EXPECT_EQ(equals.LeftArgumentPriority(), 801);
  EXPECT_EQ(equals.RightArgumentPriority(), 801);
  OperatorForm some{*FindOperator("some")->prefix}; // fxy
  EXPECT_EQ(some.LeftArgumentPriority(), 551);
  EXPECT_EQ(some.RightArgumentPriority(), 550);
  OperatorForm negate{*FindOperator("\\+")->prefix}; // fy
  EXPECT_EQ(negate.RightArgumentPriority(), 600);
}

TEST(OperatorsTest, Constants) {
  // The named priorities agree with the table.
  EXPECT_EQ(itemPriority, FindOperator(":-")->infix->priority);
  EXPECT_EQ(itemPriority, FindOperator("-->")->infix->priority);
  EXPECT_EQ(commaPriority, FindOperator(",")->infix->priority);
  EXPECT_EQ(conjunctPriority, commaPriority + 1);
  EXPECT_EQ(backquotePriority, 1500 - 120); // 22.01: `op` is yfx 120
  EXPECT_LT(lowestPriority, itemPriority);
  EXPECT_GT(primaryPriority, FindOperator(".")->infix->priority);
}

} // namespace
