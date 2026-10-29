//===-- include/merkurius/Parser/term-combinators.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_TERM_COMBINATORS_H_
#define MERKURIUS_PARSER_TERM_COMBINATORS_H_

// Combinators that convert terms into parse tree nodes, in the spirit of
// Flang's parser combinators but over the shapes of terms instead of
// sequences of tokens. A rule is a constexpr object with
//
//   using resultType = R;
//   bool Matches(const Term &) const;
//   std::optional<R> Convert(const Term &, ConvertState &) const;
//   void Expected(llvm::raw_ostream &) const;
//
// Matches looks only at the term's shape, its top functor and arity, and
// Convert is called only on a term that matches. Once a rule matches it is
// committed to: Convert reports every error below it and returns nullopt if
// there was one, so there is no backtracking and every error comes from the
// rule that applies. Apply converts a term with a rule, or reports that the
// term is not what the rule expected. Parentheses do not change a term's
// meaning, so every rule sees the term inside them.
//
// Results are flattened: a rule that matches a fixed name has the result
// std::tuple<>, and functor() concatenates the tuples of its arguments, so
// construct<T>() receives the fields of T in order.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/term.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace merkurius::parser {

struct ConvertState {
  const TokenSequence &tokens;
  ParseTreeArena &arena;
  Messages &messages;
};

// The term inside any parentheses.
const Term &StripParentheses(const Term &);

// The name of a compound term's functor or of a name, or "" for other
// terms, and the number of arguments.
llvm::StringRef FunctorName(const Term &);
std::size_t Arity(const Term &);

// Reports "expected <expected>, but found '<the term>'".
void SayExpected(ConvertState &, const Term &, llvm::StringRef expected);

namespace detail {
template <typename T> struct IsTuple : std::false_type {};
template <typename... A> struct IsTuple<std::tuple<A...>> : std::true_type {};

template <typename T> auto AsTuple(T &&x) {
  if constexpr (IsTuple<std::decay_t<T>>::value) {
    return std::forward<T>(x);
  } else {
    return std::make_tuple(std::forward<T>(x));
  }
}
template <typename T>
using TupleOf = decltype(AsTuple(std::declval<T>()));

template <typename R> std::string ExpectedText(const R &rule) {
  std::string text;
  llvm::raw_string_ostream os{text};
  rule.Expected(os);
  return text;
}
} // namespace detail

template <typename R>
std::optional<typename R::resultType> Apply(
    const R &rule, const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  if (rule.Matches(t)) {
    return rule.Convert(t, state);
  }
  SayExpected(state, t, detail::ExpectedText(rule));
  return std::nullopt;
}

// functor("name", A, B) matches a compound term name(X, Y) and converts X
// with A and Y with B; functor("name") matches the name alone.
template <typename... P> class FunctorRule {
public:
  using resultType = decltype(std::tuple_cat(
      std::declval<detail::TupleOf<typename P::resultType>>()...));
  constexpr FunctorRule(llvm::StringLiteral name, P... arguments)
      : name_{name}, arguments_{arguments...} {}
  bool Matches(const Term &t) const {
    if constexpr (sizeof...(P) == 0) {
      return llvm::isa<NameTerm>(t) && FunctorName(t) == name_;
    } else {
      return llvm::isa<CompoundTerm>(t) && FunctorName(t) == name_ &&
          Arity(t) == sizeof...(P);
    }
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    return ConvertArguments(t, state, std::index_sequence_for<P...>{});
  }
  void Expected(llvm::raw_ostream &os) const {
    os << '\'' << name_ << '\'';
    if constexpr (sizeof...(P) > 0) {
      os << '/' << sizeof...(P);
    }
  }

private:
  template <std::size_t... I>
  std::optional<resultType> ConvertArguments(const Term &t,
      ConvertState &state, std::index_sequence<I...>) const {
    if constexpr (sizeof...(P) == 0) {
      return std::tuple<>{};
    } else {
      llvm::ArrayRef<const Term *> arguments{
          llvm::cast<CompoundTerm>(t).arguments()};
      // Every argument is converted, so that the errors in each are reported;
      // the braces convert them in order.
      std::tuple<std::optional<typename P::resultType>...> results{
          Apply(std::get<I>(arguments_), *arguments[I], state)...};
      if ((std::get<I>(results).has_value() && ...)) {
        return std::tuple_cat(detail::AsTuple(*std::get<I>(results))...);
      }
      return std::nullopt;
    }
  }

  llvm::StringLiteral name_;
  std::tuple<P...> arguments_;
};

template <typename... P>
constexpr FunctorRule<P...> functor(llvm::StringLiteral name, P... arguments) {
  return {name, arguments...};
}

// withValue(v, P) converts with P and puts v before its results, to record
// which of several alternatives matched.
template <typename V, typename P> class WithValueRule {
public:
  using resultType = decltype(std::tuple_cat(std::declval<std::tuple<V>>(),
      std::declval<detail::TupleOf<typename P::resultType>>()));
  constexpr WithValueRule(V value, P rule) : value_{value}, rule_{rule} {}
  bool Matches(const Term &t) const { return rule_.Matches(t); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    if (auto result{rule_.Convert(t, state)}) {
      return std::tuple_cat(
          std::tuple<V>{value_}, detail::AsTuple(std::move(*result)));
    }
    return std::nullopt;
  }
  void Expected(llvm::raw_ostream &os) const { rule_.Expected(os); }

private:
  V value_;
  P rule_;
};

template <typename V, typename P>
constexpr WithValueRule<V, P> withValue(V value, P rule) {
  return {value, rule};
}

// construct<T>(P) makes a node of type T from the term's tokens and P's
// results.
template <typename T, typename P> class ConstructRule {
public:
  using resultType = const T *;
  constexpr explicit ConstructRule(P rule) : rule_{rule} {}
  bool Matches(const Term &t) const { return rule_.Matches(t); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    if (auto result{rule_.Convert(t, state)}) {
      return std::apply(
          [&](auto &&...fields) -> resultType {
            return state.arena.Make<T>(
                t.tokens(), std::forward<decltype(fields)>(fields)...);
          },
          detail::AsTuple(std::move(*result)));
    }
    return std::nullopt;
  }
  void Expected(llvm::raw_ostream &os) const { rule_.Expected(os); }

private:
  P rule_;
};

template <typename T, typename P>
constexpr ConstructRule<T, P> construct(P rule) {
  return ConstructRule<T, P>{rule};
}

// anyOf(P1, P2, ...) converts with the first alternative that matches. The
// alternatives must have the same result type, or anyOf<R>(...) converts
// each one's result to R, such as a std::variant of them.
template <typename R, typename... P> class AnyOfRule {
public:
  using resultType = R;
  constexpr explicit AnyOfRule(P... alternatives)
      : alternatives_{alternatives...} {}
  bool Matches(const Term &t) const {
    return std::apply(
        [&](const auto &...p) { return (p.Matches(t) || ...); },
        alternatives_);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    std::optional<resultType> result;
    std::apply(
        [&](const auto &...p) {
          // The first that matches converts; || stops after it.
          ((p.Matches(t) && (Assign(result, p.Convert(t, state)), true)) ||
              ...);
        },
        alternatives_);
    return result;
  }
  // "A, B or C".
  void Expected(llvm::raw_ostream &os) const {
    std::size_t j{0};
    std::apply(
        [&](const auto &...p) {
          ((os << (j == 0                         ? ""
                          : j + 1 == sizeof...(P) ? " or "
                                                  : ", "),
               p.Expected(os), ++j),
              ...);
        },
        alternatives_);
  }

private:
  template <typename A>
  static void Assign(std::optional<R> &to, std::optional<A> &&from) {
    if (from) {
      to = R{std::move(*from)};
    }
  }

  std::tuple<P...> alternatives_;
};

template <typename R = void, typename... P> constexpr auto anyOf(P... rules) {
  if constexpr (std::is_void_v<R>) {
    using First = typename std::tuple_element_t<0, std::tuple<P...>>::resultType;
    static_assert((std::is_same_v<First, typename P::resultType> && ...),
        "anyOf<R>() needs R when the alternatives' results differ");
    return AnyOfRule<First, P...>{rules...};
  } else {
    return AnyOfRule<R, P...>{rules...};
  }
}

// expect("a declaration", P) is P, described as "a declaration" in errors.
template <typename P> class ExpectRule {
public:
  using resultType = typename P::resultType;
  constexpr ExpectRule(llvm::StringLiteral text, P rule)
      : text_{text}, rule_{rule} {}
  bool Matches(const Term &t) const { return rule_.Matches(t); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    return rule_.Convert(t, state);
  }
  void Expected(llvm::raw_ostream &os) const { os << text_; }

private:
  llvm::StringLiteral text_;
  P rule_;
};

template <typename P>
constexpr ExpectRule<P> expect(llvm::StringLiteral text, P rule) {
  return {text, rule};
}

// commaListOf(P) converts "A, B, C", the operands of ',' terms, with P; a
// single term is a list of one.
template <typename P> class CommaListRule {
public:
  using elementType = typename P::resultType;
  using resultType = llvm::ArrayRef<elementType>;
  constexpr explicit CommaListRule(P rule) : rule_{rule} {}
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    llvm::SmallVector<elementType, 8> elements;
    bool ok{true};
    // As in mmc, both operands of a ',' may be lists, so "(a, b), c" is
    // three elements.
    auto collect{[&](const Term &term, auto &self) -> void {
      const Term &operand{StripParentheses(term)};
      if (llvm::isa<CompoundTerm>(operand) && FunctorName(operand) == "," &&
          Arity(operand) == 2) {
        for (const Term *side : llvm::cast<CompoundTerm>(operand).arguments()) {
          self(*side, self);
        }
      } else if (auto element{Apply(rule_, operand, state)}) {
        elements.push_back(std::move(*element));
      } else {
        ok = false;
      }
    }};
    collect(t, collect);
    if (!ok) {
      return std::nullopt;
    }
    return state.arena.SaveArray(llvm::ArrayRef<elementType>{elements});
  }
  void Expected(llvm::raw_ostream &os) const { rule_.Expected(os); }

private:
  P rule_;
};

template <typename P> constexpr CommaListRule<P> commaListOf(P rule) {
  return CommaListRule<P>{rule};
}

// A name, possibly module-qualified with '.' or '__', such as io.write,
// io__write or 'io'.write.
class SymNameRule {
public:
  using resultType = const SymName *;
  bool Matches(const Term &) const;
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a name"; }
};

inline constexpr SymNameRule symName;

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_TERM_COMBINATORS_H_
