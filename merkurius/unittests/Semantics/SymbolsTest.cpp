//===-- unittests/Semantics/SymbolsTest.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "gtest/gtest.h"
#include "merkurius/Semantics/symbols.h"
#include <string>

using namespace merkurius;
using namespace merkurius::semantics;

namespace {

std::string Print(const Symbol &symbol) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << symbol;
  return text;
}

TEST(SymbolsTest, OneSymbolPerKindNameAndArity) {
  ModuleSymbols symbols{"m"};
  // An abstract type in the interface, defined in the implementation.
  const Symbol &abstract{symbols.Add(
      SymbolKind::Type, "t", 1, std::monostate{}, {}, true)};
  const Symbol &defined{symbols.Add(
      SymbolKind::Type, "t", 1, std::monostate{}, {}, false)};
  EXPECT_EQ(&abstract, &defined);
  EXPECT_TRUE(defined.exported);
  // Another arity or kind is another symbol.
  symbols.Add(SymbolKind::Type, "t", 2, std::monostate{}, {}, false);
  symbols.Add(SymbolKind::Constructor, "t", 1, std::monostate{}, {}, false,
      &abstract);
  ASSERT_EQ(symbols.Lookup("t").size(), 3u);
  EXPECT_TRUE(symbols.Lookup("u").empty());
  EXPECT_EQ(symbols.symbols().size(), 3u);
  EXPECT_EQ(symbols.Lookup("t")[2]->owner, &abstract);
}

TEST(SymbolsTest, Print) {
  ModuleSymbols symbols{"list"};
  EXPECT_EQ(Print(symbols.Add(SymbolKind::Predicate, "length", 2,
                std::monostate{}, {}, true)),
      "pred list.length/2");
  EXPECT_EQ(Print(symbols.Add(SymbolKind::Function, "++", 2,
                std::monostate{}, {}, true)),
      "func list.'++'/2");
  EXPECT_EQ(Print(symbols.Add(SymbolKind::FieldUpdate, "head :=", 2,
                std::monostate{}, {}, true)),
      "field update list.'head :='/2");
  ModuleSymbols primitive{""};
  EXPECT_EQ(Print(primitive.Add(
                SymbolKind::Type, "int", 0, std::monostate{}, {}, true)),
      "type int/0");
}

} // namespace
