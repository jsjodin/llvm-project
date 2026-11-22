//===-- runtime/open.h - The open layout of terms ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The open layout of a type, which values of open(T) have (design document,
// Lowering, "Open layouts"; docs/INPROGRESS.md, Open types, step 2). It is a
// fixed function of the type's ordinary descriptor, so that the runtime can
// derive it for a type whose module never mentions open, which polymorphic
// code can receive (decision D3). Ground code never uses any of this, and a
// program that does not use open types links none of it.
//
// In the open layout every word is a tagged word, and primary tag 7 is a
// reference to a variable cell in every type, so that deref and bind need no
// type_info. A value is encoded by its type's kind:
//
// - Cells and Tuple: a pointer to a cell, as in the ordinary layout, whose
//   fields are each encoded by its own type. The functors' tags are the
//   ordinary rule's with 7 tags instead of 8: a type that needs tag 7 in
//   its ordinary layout moves one more functor under the shared tag, 6.
// - Shifted: an enum's constants, the integers of 32 bits or fewer and
//   characters, as a constant among cells is: the value, zero-extended,
//   shifted past the primary tag, so its tag is 0.
// - Boxed: int, uint, the 64-bit integers, float and string, whose words
//   may have any low bits: a pointer to a one-word cell of the ordinary
//   word, tag 0. Small ints come from a static table.
// - ClosedBox: closed(T), whose argument is a ground T in its ordinary
//   layout, which may have tag 7: a one-word box of that word.
// - AsArgument: a notag type other than closed(T), its argument's
//   encoding; an open(T)'s, T's.
// - Same: a dummy type's, 0.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_OPEN_H_
#define MERKURIUS_RUNTIME_OPEN_H_

#include "descriptor.h"

// Variable cells, deref and bind, and the trail (design document, Runtime
// system, "Support for open terms"; docs/INPROGRESS.md, Open types, step 3).
// An unbound variable is a one-word cell of the process's heap that refers
// to itself, tagged kOpenVariableTag; fields and registers hold references to
// it, and binding overwrites it, so a reference to a bound variable leads, by
// a chain, to its value. Binding never writes into a constructor's cell, so
// a cell shared with ground code is never changed. Code generation calls
// these, or inlines them, for open terms (step 4).
extern "C" {
// A new unbound variable: its tagged reference. Code generation inlines it,
// and deref, bind, box and unbox, with these as the slow paths
// (lib/CodeGen/Lowering.cpp, LowerOpenCall).
uint64_t merk_open_new_var(void);
// A one-word box of a word, as a Boxed or ClosedBox value is; and its word.
uint64_t merk_open_box(uint64_t word);
uint64_t merk_open_unbox(uint64_t box);
// The end of the chain from a word: a value, or an unbound variable's
// reference.
uint64_t merk_open_deref(uint64_t word);
// A closed(T)'s open word (a box, or a reference to a variable bound to one)
// as T's ordinary word. Code generation calls it where a closed(T) value
// that may have been unbound must be ground, as mode analysis has checked;
// one still unbound stops the program.
uint64_t merk_open_closed_value(uint64_t word);
// Binds an unbound variable, `variable` being its reference as deref gives
// it, to `value`, which must not be that reference, trailing the binding if
// an open choice point or try goal is live.
void merk_open_bind(uint64_t variable, uint64_t value);
// An open choice point or try goal begins: its mark, its place among the
// live ones, which remembers the trail's top.
uint64_t merk_open_trail_mark(void);
// Backtracking into it: the bindings since it began are undone, latest
// first.
void merk_open_trail_undo(uint64_t mark);
// It is gone, by its last alternative, its condition's success or its end,
// and with it those that began after it and are still live, which a commit
// or a soft cut has left: when no other is live, the trail is emptied. Of
// one already gone, nothing.
void merk_open_trail_release(uint64_t mark);
// The mark the next choice point would get: what a commit releases to after
// its goal's first solution, as every choice point in it is gone.
uint64_t merk_open_trail_marks(void);

// The generic walkers, by the type_info of T for open terms of type open(T):
// what code generation calls for a type it has no code of its own for, and
// for polymorphic code (design document, Runtime system, "Support for open
// terms"). The type's encodings are open.h's; a value of a type open terms
// cannot hold yet stops the program with a message.
//
// Full unification, binding variables on either side, without and with the
// occurs check; 1 if it succeeds. Without it, a cyclic term may result,
// which the walkers handle.
int32_t merk_open_unify(uint64_t typeInfo, uint64_t a, uint64_t b);
int32_t merk_open_unify_oc(uint64_t typeInfo, uint64_t a, uint64_t b);
// A ground T as an open(T): the value itself where the two layouts are the
// same, a copy otherwise.
uint64_t merk_open_to_open(uint64_t typeInfo, uint64_t value);
// An open(T) as a ground T, in *out: 0 if a variable in it is unbound or it
// is cyclic. The value itself if it has no variable cell and the layouts are
// the same, a copy otherwise, which copies a shared cell once.
int32_t merk_open_from_open(uint64_t typeInfo, uint64_t value, uint64_t *out);
// Whether the term is an unbound variable: var/1.
int32_t merk_open_is_var(uint64_t value);
// open(T)'s type_info, given open/1's descriptor and T's type_info: a cell
// that lasts as long as the program, one for each T, which code generation
// takes without a safepoint to describe a root (lib/CodeGen/MLIRGenOpen.cpp).
// T's type_info is kept too, as merk_kept_type_info_argument keeps it, so
// that a collection that moves the one it was given leaves it whole.
uint64_t merk_open_type_info(uint64_t descriptor, uint64_t argument);
// A type_info that lasts as long as the program, of a type whose arguments'
// type_infos code generation has at run time: begun with its descriptor,
// given each argument's type_info in turn, and ended. It allocates nothing
// on the heap, so it is no safepoint, as merk_open_type_info is not; an
// argument's type_info made on the heap is copied into a kept one.
uint64_t merk_kept_type_info_begin(uint64_t descriptor);
uint64_t merk_kept_type_info_argument(uint64_t key, uint64_t argument);
uint64_t merk_kept_type_info_end(uint64_t key);
// A comparison of open terms, which have no order, reached through a type
// variable: it stops the program.
[[noreturn]] void merk_open_compared(void);
}

namespace merkurius::runtime {

// A variable cell's references' primary tag, in every open layout.
constexpr Word kOpenVariableTag{7};

// Whether a word is a reference to a variable cell, bound or not.
inline bool IsVariableReference(Word word) {
  return (word & kTagMask) == kOpenVariableTag;
}
// Whether a word that deref gave is an unbound variable.
inline bool IsUnbound(Word word) { return IsVariableReference(word); }
// The tags an open layout's functors may have: 0 to 6.
constexpr unsigned kOpenTags{7};

enum class OpenEncoding : uint8_t {
  Same,
  Shifted,
  Boxed,
  Cells,
  ClosedBox,
  AsArgument,
  // An array, a closure, a type_info or a typeclass_info, which open terms
  // do not hold yet.
  Unsupported,
};

// How a value of the descriptor's type is encoded in the open layout.
OpenEncoding OpenEncodingOf(const TypeCtor &type);

// A functor's tag and secondary tag in the open layout: -1 if it has no
// secondary tag. A constant's word is the same in both layouts, tag 0.
struct OpenTag {
  uint8_t tag;
  int32_t secondary;
};

// The ordinary rule with `tags` tags (lib/CodeGen/Lowering.cpp, Layouts):
// tag 0 is the constants', if there are any; the constructors with
// arguments take the others in declaration order; when they need more than
// are left, the last is shared, and the cell's first word is the secondary
// tag. `cell` is the constructor's place among those with arguments.
constexpr OpenTag CellTag(bool constants, unsigned cells, unsigned cell,
    unsigned tags) {
  unsigned first{constants ? 1u : 0u};
  unsigned available{tags - first};
  if (cells <= available || cell + 1 < available) {
    return {static_cast<uint8_t>(first + cell), -1};
  }
  return {static_cast<uint8_t>(tags - 1),
      static_cast<int32_t>(cell - (available - 1))};
}

// A Cells type's functor's tags in the open layout, by its number among
// the descriptor's functors; a Tuple's, 0.
OpenTag OpenTagOf(const TypeCtor &type, uint32_t functor);

// Whether the type's open layout is its ordinary one, if its arguments'
// are, as the compiler found it (kOpenSame): to_open of such a value need
// not copy it.
inline bool OpenSame(const TypeCtor &type) {
  return (type.flags & kOpenSame) != 0;
}

// The static boxes of the ints from kSmallIntBoxMin to kSmallIntBoxMax, each
// a one-word cell, outside every heap; null for another int.
constexpr int64_t kSmallIntBoxMin{-128};
constexpr int64_t kSmallIntBoxMax{1023};
const Word *StaticIntBox(int64_t value);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_OPEN_H_
