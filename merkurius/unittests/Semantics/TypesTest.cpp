//===-- unittests/Semantics/TypesTest.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "gtest/gtest.h"
#include "merkurius/Semantics/types.h"

using namespace merkurius;
using namespace merkurius::semantics;

namespace {

TEST(TypesTest, Interned) {
  ModuleSymbols list{"list"}, primitive{""};
  const Symbol &listType{
      list.Add(SymbolKind::Type, "list", 1, std::monostate{}, {}, true)};
  const Symbol &intType{
      primitive.Add(SymbolKind::Type, "int", 0, std::monostate{}, {}, true)};
  TypeContext types;
  const Type *i{types.Named(intType, {})};
  EXPECT_EQ(i, types.Named(intType, {}));
  EXPECT_EQ(types.Named(listType, {i}), types.Named(listType, {i}));
  EXPECT_NE(types.Named(listType, {i}), types.Named(listType, {types.Variable(0)}));
  EXPECT_EQ(types.Variable(3), types.Variable(3));
  EXPECT_NE(types.HigherOrder(parser::PredOrFunc::Predicate,
                parser::Purity::Pure, {i}, nullptr),
      types.HigherOrder(
          parser::PredOrFunc::Predicate, parser::Purity::Impure, {i}, nullptr));
}

TEST(TypesTest, SubstituteAndPrint) {
  ModuleSymbols list{"list"}, primitive{""};
  const Symbol &listType{
      list.Add(SymbolKind::Type, "list", 1, std::monostate{}, {}, true)};
  const Symbol &intType{
      primitive.Add(SymbolKind::Type, "int", 0, std::monostate{}, {}, true)};
  TypeContext types;
  const Type *i{types.Named(intType, {})};
  const Type *t{types.Variable(0)};
  const Type *f{types.HigherOrder(parser::PredOrFunc::Function,
      parser::Purity::Pure, {types.Named(listType, {t})}, t)};
  EXPECT_EQ(TypeToString(*f), "(func(list.list(T0)) = T0)");
  const Type *g{types.Substitute(f, [&](unsigned) { return i; })};
  EXPECT_EQ(TypeToString(*g), "(func(list.list(int)) = int)");
  EXPECT_EQ(TypeToString(*types.Tuple({i, t}),
                [](unsigned v) { return "U" + std::to_string(v); }),
      "{int, U0}");
  EXPECT_EQ(TypeToString(*types.HigherOrder(parser::PredOrFunc::Predicate,
                parser::Purity::Impure, {i}, nullptr)),
      "impure pred(int)");
}

} // namespace
