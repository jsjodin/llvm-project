//===-- unittests/Common/IndirectionTest.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Common/indirection.h"
#include "gtest/gtest.h"
#include <string>
#include <variant>
#include <vector>

using merkurius::common::Indirection;

namespace {

// A recursive sum type, the use Indirection exists for: a term is an
// integer or a functor applied to terms.
struct Term;
struct Functor {
  std::string name;
  std::vector<Indirection<Term>> args;
};
struct Term {
  std::variant<int, Functor> u;
};

TEST(IndirectionTest, Recursive) {
  Functor f{"f", {}};
  f.args.emplace_back(Term{1});
  f.args.emplace_back(Term{2});
  Term t{std::move(f)};
  const Functor &g{std::get<Functor>(t.u)};
  ASSERT_EQ(g.args.size(), 2u);
  EXPECT_EQ(std::get<int>(g.args[1].value().u), 2);
}

TEST(IndirectionTest, MoveAndCopy) {
  Indirection<std::string> a{std::string{"x"}};
  Indirection<std::string> b{std::move(a)};
  EXPECT_EQ(b.value(), "x");

  Indirection<std::string, true> c{std::string{"y"}};
  Indirection<std::string, true> d{c};
  d.value() += "z";
  EXPECT_EQ(c.value(), "y");
  EXPECT_EQ(d.value(), "yz");
}

} // namespace
