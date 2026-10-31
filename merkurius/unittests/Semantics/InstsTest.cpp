//===-- unittests/Semantics/InstsTest.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Semantics/insts.h"
#include <deque>

using namespace merkurius;
using namespace merkurius::semantics;

namespace {

// Two types: abc ---> a ; b ; c, and list ---> nil ; cons(abc, list).
class Fixture : public InstTypes {
public:
  Fixture()
      : abcSymbol{module.Add(
            SymbolKind::Type, "abc", 0, std::monostate{}, {}, true)},
        listSymbol{module.Add(
            SymbolKind::Type, "list", 0, std::monostate{}, {}, true)},
        skelSymbol{module.Add(
            SymbolKind::Inst, "skel", 1, std::monostate{}, {}, true)},
        abc{types.Named(abcSymbol, {})}, list{types.Named(listSymbol, {})},
        ops{insts, *this} {
    // :- inst skel(I) ---> nil ; cons(I, skel(I)).
    const Inst *i{insts.Parameter(0, "I", nullptr)};
    const Inst *skelI{insts.Defined(skelSymbol, {i})};
    const Inst *consArguments[]{i, skelI};
    insts.SetBody(skelSymbol,
        insts.Bound(Uniqueness::Shared,
            {{Functor("nil"), {}}, {Functor("cons", 2), consArguments}}));
  }

  bool Constructors(
      const Type *type, std::vector<ConstructorTypes> &out) override {
    if (type == abc) {
      for (const char *name : {"a", "b", "c"}) {
        out.push_back({Functor(name), {}});
      }
      return true;
    }
    if (type == list) {
      out.push_back({Functor("nil"), {}});
      out.push_back({Functor("cons", 2), consTypes});
      return true;
    }
    return false;
  }

  static InstFunctor Functor(llvm::StringRef name, unsigned arity = 0) {
    return {{}, name, arity, false};
  }
  // bound(f1 ; f2 ...) of constants.
  const Inst *Constants(std::initializer_list<const char *> names) {
    std::vector<BoundAlternative> alternatives;
    for (const char *name : names) {
      alternatives.push_back({Functor(name), {}});
    }
    return insts.Bound(Uniqueness::Shared, alternatives);
  }
  const Inst *Cons(const Inst *head, const Inst *tail) {
    arguments.push_back({head, tail});
    return insts.Bound(
        Uniqueness::Shared, {{Functor("cons", 2), arguments.back()}});
  }
  const Inst *Skel(const Inst *element) {
    return insts.Defined(skelSymbol, {element});
  }

  ModuleSymbols module{"m"};
  const Symbol &abcSymbol, &listSymbol, &skelSymbol;
  TypeContext types;
  const Type *abc, *list;
  const Type *consTypes[2]{abc, list};
  InstContext insts;
  InstOps ops;
  std::deque<std::vector<const Inst *>> arguments;
};

TEST(InstsTest, Interned) {
  Fixture f;
  EXPECT_EQ(f.insts.Free(), f.insts.Free());
  EXPECT_EQ(f.Constants({"b", "a"}), f.Constants({"a", "b"}));
  EXPECT_EQ(f.Constants({"a", "a"}), f.Constants({"a"}));
  EXPECT_NE(f.insts.Ground(), f.insts.Ground(Uniqueness::Unique));
  EXPECT_EQ(f.insts.Bound(Uniqueness::Unique, {}), f.insts.NotReached());
  EXPECT_EQ(f.Skel(f.insts.Ground()), f.Skel(f.insts.Ground()));
}

TEST(InstsTest, Print) {
  Fixture f;
  EXPECT_EQ(InstToString(*f.Constants({"b", "a"})), "bound(a ; b)");
  EXPECT_EQ(InstToString(*f.Cons(f.insts.Ground(), f.Constants({"nil"}))),
      "bound(cons(ground, bound(nil)))");
  EXPECT_EQ(InstToString(*f.insts.NotReached()), "not_reached");
  EXPECT_EQ(InstToString(*f.insts.Any(Uniqueness::MostlyUnique)),
      "mostly_unique_any");
  EXPECT_EQ(InstToString(*f.Skel(f.insts.Free())), "m.skel(free)");
  const Inst *ground{f.insts.Ground()}, *free{f.insts.Free()},
      *unique{f.insts.Ground(Uniqueness::Unique)},
      *clobbered{f.insts.Ground(Uniqueness::Clobbered)};
  ArgMode modes[]{{ground, ground}, {free, ground}, {unique, clobbered}};
  EXPECT_EQ(InstToString(*f.insts.HigherOrder(parser::PredOrFunc::Predicate,
                false, modes, parser::Determinism::CcMulti)),
      "(pred(in, out, di) is cc_multi)");
  EXPECT_EQ(InstToString(*f.insts.HigherOrder(parser::PredOrFunc::Function,
                false, llvm::ArrayRef<ArgMode>{modes}.take_front(2),
                parser::Determinism::Semidet)),
      "(func(in) = out is semidet)");
  EXPECT_EQ(ArgModeToString({free, f.Constants({"a"})}), "out(bound(a))");
}

TEST(InstsTest, Orders) {
  Fixture f;
  InstOps &ops{f.ops};
  const Inst *free{f.insts.Free()}, *ground{f.insts.Ground()},
      *unique{f.insts.Ground(Uniqueness::Unique)}, *any{f.insts.Any()},
      *notReached{f.insts.NotReached()};
  // Matches: a value with the first inst can be passed for the second.
  EXPECT_TRUE(ops.Matches(free, free, f.abc));
  EXPECT_FALSE(ops.Matches(free, any, f.abc));
  EXPECT_FALSE(ops.Matches(free, ground, f.abc));
  EXPECT_TRUE(ops.Matches(ground, any, f.abc));
  EXPECT_FALSE(ops.Matches(any, ground, f.abc));
  EXPECT_TRUE(ops.Matches(unique, ground, f.abc));
  EXPECT_FALSE(ops.Matches(ground, unique, f.abc));
  EXPECT_TRUE(ops.Matches(notReached, free, f.abc));
  EXPECT_TRUE(ops.Matches(f.Constants({"a"}), ground, f.abc));
  EXPECT_TRUE(ops.Matches(f.Constants({"a"}), f.Constants({"a", "b"}), f.abc));
  EXPECT_FALSE(ops.Matches(f.Constants({"a", "c"}), f.Constants({"a", "b"}), f.abc));
  // Ground stands for each of the type's constructors.
  EXPECT_TRUE(ops.Matches(ground, f.Constants({"a", "b", "c"}), f.abc));
  EXPECT_FALSE(ops.Matches(ground, f.Constants({"a", "b"}), f.abc));
  EXPECT_FALSE(ops.Matches(ground, f.Constants({"a", "b", "c"}), nullptr));
  // Instantiatedness: a variable can go from the second inst to the first.
  EXPECT_TRUE(ops.Instantiated(ground, free, f.abc));
  EXPECT_FALSE(ops.Instantiated(free, ground, f.abc));
  EXPECT_TRUE(ops.Instantiated(ground, any, f.abc));
  EXPECT_TRUE(ops.Instantiated(any, free, f.abc));
  EXPECT_TRUE(ops.Instantiated(ground, unique, f.abc));
  EXPECT_FALSE(ops.Instantiated(unique, ground, f.abc));
  EXPECT_TRUE(ops.Instantiated(notReached, ground, f.abc));
  // Recursive insts, compared coinductively.
  EXPECT_TRUE(ops.Matches(f.Skel(ground), f.Skel(ground), f.list));
  EXPECT_TRUE(ops.Matches(f.Skel(f.Constants({"a"})), f.Skel(ground), f.list));
  EXPECT_TRUE(ops.Matches(f.Skel(ground), ground, f.list));
  EXPECT_TRUE(ops.Matches(ground, f.Skel(ground), f.list));
  EXPECT_FALSE(ops.Matches(f.Skel(free), ground, f.list));
  EXPECT_TRUE(ops.Instantiated(f.Skel(ground), f.Skel(free), f.list));
  EXPECT_FALSE(ops.Instantiated(f.Skel(free), f.Skel(ground), f.list));
  EXPECT_TRUE(ops.IsGround(f.Skel(ground), f.list));
  EXPECT_TRUE(ops.HasFree(f.Skel(free), f.list));
  EXPECT_FALSE(ops.HasFree(f.Skel(ground), f.list));
}

// normalise (Overton, definition 3.4.1): a ground inst becomes ground of
// the least unique uniqueness of its parts; another is kept.
TEST(InstsTest, Normalise) {
  Fixture f;
  InstOps &ops{f.ops};
  const Inst *free{f.insts.Free()}, *ground{f.insts.Ground()},
      *unique{f.insts.Ground(Uniqueness::Unique)};
  const Inst *uniqueA{f.insts.Bound(Uniqueness::Unique, {{f.Functor("a"), {}}})};
  EXPECT_EQ(ops.Normalise(uniqueA, f.abc), unique);
  EXPECT_EQ(ops.Normalise(f.Constants({"a", "b"}), f.abc), ground);
  EXPECT_EQ(ops.Normalise(unique, f.abc), unique);
  EXPECT_EQ(ops.Normalise(free, f.abc), free);
  EXPECT_EQ(ops.Normalise(f.insts.NotReached(), f.abc), f.insts.NotReached());
  const Inst *arguments[]{ground, f.insts.Bound(Uniqueness::Unique, {{f.Functor("nil"), {}}})};
  const Inst *mixed{f.insts.Bound(
      Uniqueness::Unique, {{f.Functor("cons", 2), arguments}})};
  EXPECT_EQ(ops.Normalise(mixed, f.list), ground);
  const Inst *partial{f.Cons(f.Constants({"a"}), free)};
  EXPECT_EQ(ops.Normalise(partial, f.list), partial);
}

TEST(InstsTest, Merge) {
  Fixture f;
  InstOps &ops{f.ops};
  const Inst *free{f.insts.Free()}, *ground{f.insts.Ground()},
      *unique{f.insts.Ground(Uniqueness::Unique)}, *any{f.insts.Any()};
  EXPECT_EQ(ops.Merge(f.Constants({"a"}), f.Constants({"b"}), f.abc),
      f.Constants({"a", "b"}));
  EXPECT_EQ(ops.Merge(f.insts.NotReached(), f.Constants({"b"}), f.abc),
      f.Constants({"b"}));
  EXPECT_EQ(ops.Merge(free, free, f.abc), free);
  EXPECT_EQ(ops.Merge(free, ground, f.abc), nullptr);
  EXPECT_EQ(ops.Merge(f.Constants({"a"}), ground, f.abc), ground);
  EXPECT_EQ(ops.Merge(unique, ground, f.abc), ground);
  EXPECT_EQ(ops.Merge(ground, any, f.abc), any);
  // A bound inst with a free part merges argument by argument.
  const Inst *consFree{f.Cons(free, f.Constants({"nil"}))};
  EXPECT_EQ(ops.Merge(consFree, f.Constants({"nil"}), f.list),
      f.insts.Bound(Uniqueness::Shared,
          {{Fixture::Functor("cons", 2), consFree->functors[0].arguments},
              {Fixture::Functor("nil"), {}}}));
  EXPECT_EQ(ops.Merge(consFree, ground, f.list), nullptr);
  // A recursive result is a generated inst.
  const Inst *merged{
      ops.Merge(f.Skel(f.Constants({"a"})), f.Skel(f.Constants({"b"})), f.list)};
  ASSERT_NE(merged, nullptr);
  EXPECT_TRUE(ops.Matches(merged, f.Skel(f.Constants({"a", "b"})), f.list));
  EXPECT_TRUE(ops.Matches(f.Skel(f.Constants({"a", "b"})), merged, f.list));
}

TEST(InstsTest, Unify) {
  Fixture f;
  InstOps &ops{f.ops};
  const Inst *free{f.insts.Free()}, *ground{f.insts.Ground()},
      *unique{f.insts.Ground(Uniqueness::Unique)}, *any{f.insts.Any()};
  EXPECT_EQ(ops.Unify(free, ground, f.abc), ground);
  EXPECT_EQ(ops.Unify(unique, ground, f.abc), ground);
  EXPECT_EQ(ops.Unify(any, ground, f.abc), ground);
  EXPECT_EQ(ops.Unify(f.Constants({"a", "b"}), f.Constants({"b", "c"}), f.abc),
      f.Constants({"b"}));
  EXPECT_EQ(ops.Unify(f.Constants({"a"}), f.Constants({"b"}), f.abc),
      f.insts.NotReached());
  EXPECT_EQ(ops.Unify(f.Constants({"a"}), ground, f.abc), f.Constants({"a"}));
  EXPECT_EQ(ops.Unify(f.Cons(free, f.Constants({"nil"})), ground, f.list),
      f.Cons(ground, f.Constants({"nil"})));
  // An alternative with a not_reached argument is dropped.
  EXPECT_EQ(ops.Unify(f.Cons(f.Constants({"a"}), ground),
                f.Cons(f.Constants({"b"}), ground), f.list),
      f.insts.NotReached());
  const Inst *unified{ops.Unify(f.Skel(free), f.Skel(ground), f.list)};
  ASSERT_NE(unified, nullptr);
  EXPECT_TRUE(ops.Matches(unified, f.Skel(ground), f.list));
  EXPECT_TRUE(ops.IsGround(unified, f.list));
  const Inst *pred{f.insts.HigherOrder(
      parser::PredOrFunc::Predicate, false, {}, parser::Determinism::Det)};
  EXPECT_EQ(ops.Unify(pred, ground, nullptr), pred);
  EXPECT_EQ(ops.Unify(pred,
                f.insts.HigherOrder(parser::PredOrFunc::Predicate, false, {},
                    parser::Determinism::Semidet),
                nullptr),
      nullptr);
}

TEST(InstsTest, Share) {
  Fixture f;
  const Inst *unique{f.insts.Ground(Uniqueness::Unique)};
  EXPECT_EQ(f.insts.Share(unique), f.insts.Ground());
  EXPECT_EQ(f.insts.Share(f.Cons(unique, f.insts.Free())),
      f.Cons(f.insts.Ground(), f.insts.Free()));
  EXPECT_EQ(f.insts.Share(f.insts.Ground(Uniqueness::Clobbered)),
      f.insts.Ground(Uniqueness::Clobbered));
}

} // namespace
