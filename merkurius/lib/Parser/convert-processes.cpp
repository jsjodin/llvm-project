//===-- lib/Parser/convert-processes.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The converter's rule for process declarations, an extension (design
// document, "Processes and mailboxes"), and the predicates each expands to,
// made as the terms they would be written as and converted as items:
//
//   :- process counter(int) receives counter_msg.
//
//   :- pred counter(int::in, process.mailbox(counter_msg)::in,
//       io.io::di, io.io::uo) is det.
//   :- pred spawn_counter(int::in, process.mailbox(counter_msg)::out,
//       io.io::di, io.io::uo) is det.
//   spawn_counter(A1, Mailbox, IO0, IO) :-
//       process.spawn((pred(Self::in, Self0::di, Self1::uo) is det :-
//           m.counter(A1, Self, Self0, Self1)), Mailbox, IO0, IO).
//
// where m is the module. The names are qualified, so that the expansion
// means the same whatever else the module declares; the module must import
// io and process, as the kind's clauses use them.

#include "convert-rules.h"
#include "merkurius/Parser/convert.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace merkurius::parser {

namespace {

// Makes the terms of a declaration's expansion in the item's arena, each
// with the declaration's tokens, so that an error in one is reported at the
// declaration.
class TermMaker {
public:
  TermMaker(TermArena &arena, TokenRange tokens)
      : arena_{arena}, tokens_{tokens} {}

  const NameTerm *Name(llvm::StringRef name) {
    return arena_.Make<NameTerm>(tokens_, arena_.Save(name));
  }
  const Term *Compound(llvm::StringRef functor,
      llvm::ArrayRef<const Term *> arguments,
      CompoundForm form = CompoundForm::Canonical) {
    if (arguments.empty()) {
      return Name(functor);
    }
    return arena_.Make<CompoundTerm>(
        tokens_, Name(functor), arena_.Save(arguments), form);
  }
  const Term *Infix(llvm::StringRef op, const Term *left, const Term *right) {
    return Compound(op, {left, right}, CompoundForm::Infix);
  }
  const Term *Prefix(llvm::StringRef op, const Term *operand) {
    return Compound(op, {operand}, CompoundForm::Prefix);
  }
  const Term *Variable(llvm::StringRef name, unsigned number) {
    return arena_.Make<VariableTerm>(tokens_, arena_.Save(name), number);
  }
  // Type::Mode.
  const Term *Moded(const Term *type, llvm::StringRef mode) {
    return Infix("::", type, Name(mode));
  }
  // The term qualified by the module `qualifiers`, the outermost first:
  // a.b.t as '.'('.'(a, b), t), as the reader reads it.
  const Term *Qualified(
      llvm::ArrayRef<llvm::StringRef> qualifiers, const Term *term) {
    if (qualifiers.empty()) {
      return term;
    }
    const Term *module{Name(qualifiers.front())};
    for (llvm::StringRef qualifier : qualifiers.drop_front()) {
      module = Infix(".", module, Name(qualifier));
    }
    return Infix(".", module, term);
  }
  const Term *IoType() { return Qualified({"io"}, Name("io")); }
  const Term *MailboxType(const Term *message) {
    return Qualified({"process"}, Compound("mailbox", {message}));
  }

private:
  TermArena &arena_;
  TokenRange tokens_;
};

// :- pred Name(Arguments) is det.
const Term &PredDeclTerm(
    TermMaker &make, llvm::StringRef name, llvm::ArrayRef<const Term *> args) {
  return *make.Prefix(":-",
      make.Prefix("pred",
          make.Infix("is", make.Compound(name, args), make.Name("det"))));
}

// The kind's predicate: its arguments, a mailbox for each message type, and
// the io state.
const PredDecl *KindDecl(TermMaker &make, llvm::StringRef name,
    llvm::ArrayRef<const Term *> arguments,
    llvm::ArrayRef<const Term *> received, ConvertState &state) {
  llvm::SmallVector<const Term *, 8> args;
  for (const Term *argument : arguments) {
    args.push_back(make.Moded(argument, "in"));
  }
  for (const Term *message : received) {
    args.push_back(make.Moded(make.MailboxType(message), "in"));
  }
  args.push_back(make.Moded(make.IoType(), "di"));
  args.push_back(make.Moded(make.IoType(), "uo"));
  std::optional<Item> item{ConvertItem(PredDeclTerm(make, name, args), state)};
  return item ? std::get<const PredDecl *>(*item) : nullptr;
}

// spawn_Kind's declaration, for a kind that receives one message type.
const PredDecl *SpawnDecl(TermMaker &make, llvm::StringRef spawnName,
    llvm::ArrayRef<const Term *> arguments, const Term *message,
    ConvertState &state) {
  llvm::SmallVector<const Term *, 8> args;
  for (const Term *argument : arguments) {
    args.push_back(make.Moded(argument, "in"));
  }
  args.push_back(make.Moded(make.MailboxType(message), "out"));
  args.push_back(make.Moded(make.IoType(), "di"));
  args.push_back(make.Moded(make.IoType(), "uo"));
  std::optional<Item> item{
      ConvertItem(PredDeclTerm(make, spawnName, args), state)};
  return item ? std::get<const PredDecl *>(*item) : nullptr;
}

// spawn_Kind's clause, which spawns a process that calls the kind's
// predicate, module-qualified, with the arguments and its mailbox.
const FactOrRule *SpawnClause(TermMaker &make, llvm::StringRef name,
    llvm::StringRef spawnName, std::size_t arity, ConvertState &state) {
  auto count{static_cast<unsigned>(arity)};
  llvm::SmallVector<const Term *, 8> head, call;
  for (unsigned i{0}; i < count; ++i) {
    const Term *variable{make.Variable("A" + std::to_string(i + 1), i)};
    head.push_back(variable);
    call.push_back(variable);
  }
  const Term *mailbox{make.Variable("Mailbox", count)};
  const Term *io0{make.Variable("IO0", count + 1)};
  const Term *io{make.Variable("IO", count + 2)};
  const Term *self{make.Variable("Self", count + 3)};
  const Term *self0{make.Variable("Self0", count + 4)};
  const Term *self1{make.Variable("Self1", count + 5)};
  head.append({mailbox, io0, io});
  call.append({self, self0, self1});
  llvm::SmallVector<llvm::StringRef, 4> module;
  if (state.module) {
    module.append(
        state.module->qualifiers.begin(), state.module->qualifiers.end());
    module.push_back(state.module->name);
  }
  const Term *lambda{make.Infix(":-",
      make.Infix("is",
          make.Compound("pred",
              {make.Moded(self, "in"), make.Moded(self0, "di"),
                  make.Moded(self1, "uo")}),
          make.Name("det")),
      make.Qualified(module, make.Compound(name, call)))};
  const Term *body{make.Qualified(
      {"process"}, make.Compound("spawn", {lambda, mailbox, io0, io}))};
  const Term *clause{make.Infix(":-", make.Compound(spawnName, head), body)};
  // The clause's variables are its own, not the declaration's.
  const Term *item{state.item};
  unsigned variables{state.variables};
  state.item = clause;
  state.variables = count + 6;
  std::optional<Item> converted{ConvertItem(*clause, state)};
  state.item = item;
  state.variables = variables;
  return converted ? std::get<const FactOrRule *>(*converted) : nullptr;
}

// The types of `terms`, or none after an error, which is reported.
std::optional<llvm::ArrayRef<Type>> ConvertTypes(
    llvm::ArrayRef<const Term *> terms, ConvertState &state) {
  llvm::SmallVector<Type, 4> types;
  bool ok{true};
  for (const Term *term : terms) {
    if (std::optional<Type> converted{Apply(type, *term, state)}) {
      types.push_back(*converted);
    } else {
      ok = false;
    }
  }
  if (!ok) {
    return std::nullopt;
  }
  return state.arena.SaveArray(llvm::ArrayRef<Type>{types});
}

} // namespace

std::optional<const ProcessDecl *> ProcessRule::Convert(
    const Term &t, ConvertState &state) const {
  // Without -fprocess-declarations, process is not an operator, and
  // :- process(...) is no declaration, as in mmc.
  if (!state.processDeclarations || !state.terms) {
    SayExpected(state, t, "a declaration");
    return std::nullopt;
  }
  const Term *body{
      &StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
  bool ok{true}, onGpu{false};
  if (const CompoundTerm *on{AsCompound(*body, "on", 2)}) {
    const Term &place{StripParentheses(*on->arguments()[1])};
    if (!llvm::isa<NameTerm>(place) || FunctorName(place) != "gpu") {
      SayExpected(state, place, "gpu, the place where the process kind runs");
      ok = false;
    }
    onGpu = true;
    body = &StripParentheses(*on->arguments()[0]);
  }
  const CompoundTerm *receives{AsCompound(*body, "receives", 2)};
  if (!receives) {
    SayExpected(state, *body,
        "a process kind, followed by receives and its message types");
    return std::nullopt;
  }
  const Term &kind{StripParentheses(*receives->arguments()[0])};
  const auto *kindCompound{llvm::dyn_cast<CompoundTerm>(&kind)};
  if (!llvm::isa<NameTerm>(kind) &&
      !(kindCompound && kindCompound->form() == CompoundForm::Canonical)) {
    SayExpected(state, kind,
        "a process kind, a name with its argument types, not module-qualified");
    return std::nullopt;
  }
  llvm::ArrayRef<const Term *> argumentTerms;
  if (kindCompound) {
    argumentTerms = kindCompound->arguments();
  }
  // Several message types are separated by ','.
  llvm::SmallVector<const Term *, 2> received;
  const Term *rest{receives->arguments()[1]};
  while (const CompoundTerm *comma{AsCompound(*rest, ",", 2)}) {
    received.push_back(comma->arguments()[0]);
    rest = comma->arguments()[1];
  }
  received.push_back(rest);
  std::optional<llvm::ArrayRef<Type>> arguments{
      ConvertTypes(argumentTerms, state)};
  std::optional<llvm::ArrayRef<Type>> messages{ConvertTypes(received, state)};
  if (!ok || !arguments || !messages) {
    return std::nullopt;
  }
  llvm::StringRef name{state.arena.Save(FunctorName(kind))};
  std::string spawnName{"spawn_" + name.str()};
  TermMaker make{*state.terms, t.tokens()};
  const PredDecl *kindDecl{
      KindDecl(make, name, argumentTerms, received, state)};
  const PredDecl *spawnDecl{nullptr};
  const FactOrRule *spawnClause{nullptr};
  if (received.size() == 1) {
    spawnDecl = SpawnDecl(make, spawnName, argumentTerms, received[0], state);
    spawnClause =
        SpawnClause(make, name, spawnName, argumentTerms.size(), state);
  }
  if (!kindDecl || (received.size() == 1 && (!spawnDecl || !spawnClause))) {
    return std::nullopt;
  }
  return state.arena.Make<ProcessDecl>(t.tokens(), name, *arguments, *messages,
      onGpu, kindDecl, spawnDecl, spawnClause);
}

} // namespace merkurius::parser
