//===-- lib/Parser/term-reader.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/operators.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include <algorithm>
#include <cstdlib>

namespace merkurius::parser {

namespace {

void AppendUTF8(std::string &out, std::uint32_t code) {
  char buffer[UNI_MAX_UTF8_BYTES_PER_CODE_POINT];
  char *end{buffer};
  if (llvm::ConvertCodePointToUTF8(code, end)) {
    out.append(buffer, end);
  }
}

} // namespace

// The escapes are those the lexer accepts; it has already reported any
// that are invalid, which are kept as written.
std::string UnquoteSpelling(llvm::StringRef spelling) {
  if (spelling.size() < 2 || (spelling[0] != '\'' && spelling[0] != '"')) {
    return spelling.str();
  }
  char quote{spelling[0]};
  llvm::StringRef text{spelling.drop_front()};
  if (text.ends_with(llvm::StringRef(&quote, 1))) {
    text = text.drop_back();
  }
  std::string out;
  for (std::size_t j{0}; j < text.size(); ++j) {
    char ch{text[j]};
    if (ch == quote && j + 1 < text.size() && text[j + 1] == quote) {
      out += quote;
      ++j;
      continue;
    }
    if (ch != '\\' || j + 1 == text.size()) {
      out += ch;
      continue;
    }
    char escape{text[++j]};
    switch (escape) {
    case 'a':
      out += '\a';
      continue;
    case 'b':
      out += '\b';
      continue;
    case 'r':
      out += '\r';
      continue;
    case 'f':
      out += '\f';
      continue;
    case 't':
      out += '\t';
      continue;
    case 'n':
      out += '\n';
      continue;
    case 'v':
      out += '\v';
      continue;
    case '\\':
    case '\'':
    case '"':
      out += escape;
      continue;
    case '\n': // a continuation line
      continue;
    case '\r':
      if (j + 1 < text.size() && text[j + 1] == '\n') {
        ++j;
        continue;
      }
      break;
    case 'u':
    case 'U': {
      std::size_t count{escape == 'u' ? 4u : 8u};
      std::uint32_t code;
      if (!text.substr(j + 1, count).getAsInteger(16, code) &&
          j + count < text.size()) {
        AppendUTF8(out, code);
        j += count;
        continue;
      }
      break;
    }
    case 'x':
    default: {
      bool hex{escape == 'x'};
      std::size_t digits{hex ? j + 1 : j};
      std::size_t close{text.find('\\', digits)};
      std::uint32_t code;
      if ((hex || (escape >= '0' && escape <= '7')) &&
          close != llvm::StringRef::npos && close > digits &&
          !text.slice(digits, close).getAsInteger(hex ? 16 : 8, code)) {
        AppendUTF8(out, code);
        j = close;
        continue;
      }
      break;
    }
    }
    out += '\\';
    out += escape;
  }
  return out;
}

TermReader::TermReader(const TokenSequence &tokens, TermArena &arena,
    Messages &messages, unsigned maxDepth)
    : tokens_{tokens}, arena_{arena}, messages_{messages}, maxDepth_{maxDepth} {
}

const Token &TermReader::Peek(std::size_t ahead) const {
  std::size_t j{std::min<std::size_t>(at_ + ahead, tokens_.tokens.size() - 1)};
  return tokens_.tokens[j];
}

// The tokens that end an argument, element, parenthesized term or item.
bool TermReader::AtTerminator(std::size_t ahead) const {
  switch (PeekKind(ahead)) {
  case TokenKind::Close:
  case TokenKind::CloseList:
  case TokenKind::CloseCurly:
  case TokenKind::HtSep:
  case TokenKind::End:
  case TokenKind::Eof:
    return true;
  case TokenKind::Comma:
    return !commaIsOperator_;
  default:
    return false;
  }
}

// Whether the tokens from the backquote `ahead` are `Name`, `Module.Name`
// or `Variable`. As in mmc, a backquote that does not begin one of these is
// not an operator; where a term may begin, it is the name '`'.
bool TermReader::AtBackquoteOperator(std::size_t ahead) const {
  TokenKind kind{PeekKind(ahead + 1)};
  if (kind != TokenKind::Name && kind != TokenKind::Variable) {
    return false;
  }
  ahead += 2;
  while (kind == TokenKind::Name && Peek(ahead).spelling == "." &&
      PeekKind(ahead + 1) == TokenKind::Name) {
    ahead += 2;
  }
  return PeekKind(ahead) == TokenKind::Backquote;
}

bool TermReader::IsInfixOperatorAt(std::size_t ahead) const {
  const Token &token{Peek(ahead)};
  if (token.kind == TokenKind::Comma) {
    return commaIsOperator_;
  }
  const OperatorDefinition *op{GetOperator(token)};
  return op && op->infix;
}

const OperatorDefinition *TermReader::GetOperator(const Token &token) const {
  if (token.kind != TokenKind::Name) {
    return nullptr;
  }
  return FindOperator(UnquoteSpelling(token.spelling));
}

// Whether the token after a prefix operator can begin its operand. A name
// that is only an infix operator cannot, unless it is a compound term's
// functor, so in "- = X" the '-' is not applied to anything.
bool TermReader::CanStartPrefixOperand() const {
  if (AtTerminator(1)) {
    return false;
  }
  switch (PeekKind(1)) {
  case TokenKind::Comma:
  case TokenKind::Error:
    return false;
  case TokenKind::Name:
    if (const OperatorDefinition * op{GetOperator(Peek(1))};
        op && op->infix && !op->prefix) {
      return PeekKind(2) == TokenKind::OpenCt;
    }
    return true;
  default:
    return true;
  }
}

template <typename... A>
void TermReader::Say(std::uint32_t first, A &&...args) {
  const Token &begin{tokens_.tokens[first]};
  const Token &end{tokens_.tokens[std::max(first, at_ - 1)]};
  messages_.Say(CharBlock{begin.spelling.begin(), end.spelling.end()},
      std::forward<A>(args)...);
}

void TermReader::SayUnexpected(llvm::StringRef expected) {
  if (failed_) {
    return;
  }
  failed_ = true;
  const Token &token{Peek()};
  if (token.kind == TokenKind::Error) {
    return; // the lexer has reported it
  }
  if (token.kind == TokenKind::Eof) {
    messages_.Say(token.spelling,
        "expected %s, but found the end of the file"_err_en_US, expected.str());
  } else {
    messages_.Say(token.spelling, "expected %s, but found '%s'"_err_en_US,
        expected.str(), token.spelling.ToString());
  }
}

void TermReader::SkipItem() {
  while (PeekKind() != TokenKind::Eof) {
    if (tokens_.tokens[Next()].kind == TokenKind::End) {
      return;
    }
  }
}

std::optional<Item> TermReader::ReadItem() {
  if (PeekKind() == TokenKind::Eof) {
    return std::nullopt;
  }
  std::uint32_t first{at_};
  failed_ = false;
  commaIsOperator_ = true;
  argumentStart_ = at_;
  variableNumbers_.clear();
  variables_ = 0;
  Parsed parsed{Parse(itemPriority)};
  if (!failed_ && PeekKind() == TokenKind::End) {
    Next();
  } else {
    SayUnexpected("an operator or '.'");
    SkipItem();
  }
  return Item{failed_ ? nullptr : parsed.term, Range(first), variables_};
}

// Reads a term whose priority is at least minPriority, by precedence
// climbing: a primary term, then each infix operator that binds less
// tightly than what came before but no less than minPriority.
TermReader::Parsed TermReader::Parse(int minPriority) {
  if (depth_ == maxDepth_) {
    if (!failed_) {
      failed_ = true;
      messages_.Say(Peek().spelling,
          "the term nests more than %u levels deep"_err_en_US, maxDepth_);
    }
    return {};
  }
  ++depth_;
  Parsed result{ParseOperators(minPriority)};
  --depth_;
  return result;
}

TermReader::Parsed TermReader::ParseOperators(int minPriority) {
  Parsed left{ParsePrimary(minPriority)};
  while (!failed_) {
    const Token &token{Peek()};
    if (token.kind == TokenKind::Backquote && AtBackquoteOperator(0)) {
      if (backquotePriority < minPriority ||
          left.priority < backquotePriority) {
        break;
      }
      Next();
      const Term *op{ParseBackquoteOperator()};
      if (failed_) {
        break;
      }
      Parsed right{Parse(backquotePriority + 1)};
      if (failed_) {
        break;
      }
      left = {arena_.Make<BackquoteTerm>(
                  Range(left.term->tokens().first), left.term, op, right.term),
          backquotePriority};
      continue;
    }
    std::optional<OperatorForm> form;
    if (token.kind == TokenKind::Comma) {
      if (commaIsOperator_) {
        form = FindOperator(",")->infix;
      }
    } else if (const OperatorDefinition * op{GetOperator(token)}) {
      form = op->infix;
    }
    if (!form || form->priority < minPriority ||
        left.priority < form->LeftArgumentPriority()) {
      break;
    }
    std::uint32_t opToken{Next()};
    Parsed right{Parse(form->RightArgumentPriority())};
    if (failed_) {
      break;
    }
    // ',' is a token of its own, but the functor is the name ','.
    const NameTerm *functor{token.kind == TokenKind::Comma
            ? arena_.Make<NameTerm>(TokenRange{opToken, opToken}, ",")
            : MakeName(opToken)};
    const Term *arguments[]{left.term, right.term};
    left = {arena_.Make<CompoundTerm>(Range(left.term->tokens().first), functor,
                arena_.Save(arguments), CompoundForm::Infix),
        form->priority};
  }
  return left;
}

// A primary term. As in mmc, any primary term other than a name followed
// by open_ct is a higher-order term: 1(X), "s"(X) and [A](X) as well as
// P(X). The exceptions are [] and {}, which are then names: [](X) is the
// compound term '[]'(X).
TermReader::Parsed TermReader::ParsePrimary(int minPriority) {
  std::uint32_t first{at_};
  switch (PeekKind()) {
  case TokenKind::Variable:
    return {ParseApplications(MakeVariable(Next()), first), primaryPriority};
  case TokenKind::Integer:
  case TokenKind::Float:
    return {ParseApplications(MakeLiteral(first, Next(), false), first),
        primaryPriority};
  case TokenKind::String: {
    llvm::StringRef spelling{tokens_.tokens[Next()].spelling};
    const Term *string{arena_.Make<StringTerm>(
        Range(first), arena_.Save(UnquoteSpelling(spelling)))};
    return {ParseApplications(string, first), primaryPriority};
  }
  case TokenKind::ImplementationDefinedLiteral: {
    llvm::StringRef spelling{tokens_.tokens[Next()].spelling};
    const Term *literal{arena_.Make<ImplementationDefinedLiteralTerm>(
        Range(first), spelling.drop_front())};
    return {ParseApplications(literal, first), primaryPriority};
  }
  case TokenKind::Name:
    return ParseName(minPriority);
  case TokenKind::Open:
  case TokenKind::OpenCt: {
    Next();
    Parsed inner{ParseNested(lowestPriority, /*commaIsOperator=*/true)};
    if (failed_) {
      return {};
    }
    if (PeekKind() != TokenKind::Close) {
      SayUnexpected("an operator or ')'");
      return {};
    }
    Next();
    const Term *paren{arena_.Make<ParenthesizedTerm>(Range(first), inner.term)};
    return {ParseApplications(paren, first), primaryPriority};
  }
  case TokenKind::OpenList: {
    Next();
    llvm::SmallVector<const Term *, 8> elements;
    const Term *tail{nullptr};
    if (PeekKind() == TokenKind::CloseList) {
      Next();
      if (PeekKind() == TokenKind::OpenCt) {
        return ParseEmptyBracketsCall("[]", first);
      }
    } else if (!ParseArguments(TokenKind::CloseList, elements, &tail)) {
      return {};
    }
    const Term *list{
        arena_.Make<ListTerm>(Range(first), arena_.Save(elements), tail)};
    return {ParseApplications(list, first), primaryPriority};
  }
  case TokenKind::OpenCurly: {
    // As in mmc, the contents are one term in which ',' is an operator; the
    // elements are the operands of its top-level ',' operators, looking
    // through parentheses, so {a :- b, c} has one element.
    Next();
    if (PeekKind() == TokenKind::CloseCurly) {
      Next();
      if (PeekKind() == TokenKind::OpenCt) {
        return ParseEmptyBracketsCall("{}", first);
      }
      return {ParseApplications(arena_.Make<TupleTerm>(Range(first),
                                    llvm::ArrayRef<const Term *>{}),
                  first),
          primaryPriority};
    }
    Parsed contents{ParseNested(lowestPriority, /*commaIsOperator=*/true)};
    if (failed_) {
      return {};
    }
    if (PeekKind() != TokenKind::CloseCurly) {
      SayUnexpected("an operator or '}'");
      return {};
    }
    Next();
    llvm::SmallVector<const Term *, 8> elements;
    const Term *rest{contents.term};
    while (true) {
      const Term *spine{rest};
      while (const auto *paren{llvm::dyn_cast<ParenthesizedTerm>(spine)}) {
        spine = paren->inner();
      }
      const auto *comma{llvm::dyn_cast<CompoundTerm>(spine)};
      if (!comma || comma->form() != CompoundForm::Infix ||
          comma->functor()->value() != ",") {
        break;
      }
      elements.push_back(comma->arguments()[0]);
      rest = comma->arguments()[1];
    }
    elements.push_back(rest);
    const Term *tuple{
        arena_.Make<TupleTerm>(Range(first), arena_.Save(elements))};
    return {ParseApplications(tuple, first), primaryPriority};
  }
  case TokenKind::Backquote:
    // As in mmc, a backquote where a term begins is the name '`'.
    return {MakeName(Next()), primaryPriority};
  case TokenKind::Directive: {
    // The annotated goal is one conjunct, and has its priority.
    Next();
    Parsed annotated{Parse(std::max(minPriority, conjunctPriority))};
    if (failed_) {
      return {};
    }
    return {arena_.Make<DirectiveTerm>(Range(first), first, annotated.term),
        annotated.priority};
  }
  default:
    SayUnexpected("a term");
    return {};
  }
}

// [](Args) or {}(Args), at the open_ct after the brackets.
TermReader::Parsed TermReader::ParseEmptyBracketsCall(
    llvm::StringRef name, std::uint32_t first) {
  const NameTerm *functor{
      arena_.Make<NameTerm>(TokenRange{first, first + 1}, name)};
  Next();
  llvm::SmallVector<const Term *, 8> arguments;
  if (!ParseArguments(TokenKind::Close, arguments)) {
    return {};
  }
  const Term *compound{arena_.Make<CompoundTerm>(
      Range(first), functor, arena_.Save(arguments), CompoundForm::Canonical)};
  return {ParseApplications(compound, first), primaryPriority};
}

// Whether an integer or float token is an unsigned integer, which has a
// 'u' suffix; a character code such as 0'u has none.
static bool IsUnsignedLiteral(const Token &token) {
  return token.kind == TokenKind::Integer &&
      !token.spelling.starts_with("0'") && token.spelling.contains('u');
}

// A name: a compound term's functor, a negative number, a prefix operator
// and its operands, or a constant.
TermReader::Parsed TermReader::ParseName(int minPriority) {
  std::uint32_t first{at_};
  const Token &token{Peek()};
  if (PeekKind(1) == TokenKind::OpenCt) {
    const NameTerm *functor{MakeName(Next())};
    Next();
    llvm::SmallVector<const Term *, 8> arguments;
    if (!ParseArguments(TokenKind::Close, arguments)) {
      return {};
    }
    const Term *compound{arena_.Make<CompoundTerm>(Range(first), functor,
        arena_.Save(arguments), CompoundForm::Canonical)};
    return {ParseApplications(compound, first), primaryPriority};
  }
  // As in mmc, '-' before a number is its sign, whatever separates them
  // and however '-' is written, so "- 1 ^ 2" is (-1) ^ 2. Unsigned
  // literals have no sign: -1u is '-'(1u).
  if ((PeekKind(1) == TokenKind::Integer || PeekKind(1) == TokenKind::Float) &&
      !IsUnsignedLiteral(Peek(1)) && UnquoteSpelling(token.spelling) == "-") {
    // Unlike other primary terms, a negative number cannot be applied.
    Next();
    return {MakeLiteral(first, Next(), true), primaryPriority};
  }
  const OperatorDefinition *op{GetOperator(token)};
  if (op && op->prefix && CanStartPrefixOperand()) {
    OperatorForm form{*op->prefix};
    if (form.priority < minPriority) {
      Next();
      Say(first,
          "the prefix operator '%s' binds too loosely to be used here without parentheses"_err_en_US,
          UnquoteSpelling(token.spelling));
      failed_ = true;
      return {};
    }
    const NameTerm *functor{MakeName(Next())};
    llvm::SmallVector<const Term *, 2> arguments;
    if (form.specifier == Specifier::FXY) {
      arguments.push_back(Parse(form.LeftArgumentPriority()).term);
      if (failed_) {
        return {};
      }
    }
    arguments.push_back(Parse(form.RightArgumentPriority()).term);
    if (failed_) {
      return {};
    }
    return {arena_.Make<CompoundTerm>(Range(first), functor,
                arena_.Save(arguments), CompoundForm::Prefix),
        form.priority};
  }
  // An operator used as a constant must be parenthesized or begin an
  // argument, element or item (reference manual, "Builtin operators"). As
  // in mmc, it may be followed there by an infix operator, as in "; -> a",
  // unless it is also a prefix operator, as in "- = a".
  // Before ',' as an operator, any operator may be a constant there.
  bool constantAllowed{first == argumentStart_ &&
      (AtTerminator(1) || PeekKind(1) == TokenKind::Comma ||
          (op && !op->prefix &&
              (IsInfixOperatorAt(1) ||
                  (PeekKind(1) == TokenKind::Backquote &&
                      AtBackquoteOperator(1)))))};
  if (op && !constantAllowed) {
    Next();
    Say(first,
        "the operator '%s' must be in parentheses to be used as an operand"_err_en_US,
        UnquoteSpelling(token.spelling));
    failed_ = true;
    return {};
  }
  return {MakeName(Next()), primaryPriority};
}

// A parenthesized term, argument or element, with its own treatment of ','.
TermReader::Parsed TermReader::ParseNested(int priority, bool commaIsOperator) {
  bool savedComma{commaIsOperator_};
  std::uint32_t savedStart{argumentStart_};
  commaIsOperator_ = commaIsOperator;
  argumentStart_ = at_;
  Parsed parsed{Parse(priority)};
  commaIsOperator_ = savedComma;
  argumentStart_ = savedStart;
  return parsed;
}

// An argument or element; the caller handles what follows it.
const Term *TermReader::ParseArgument() {
  return ParseNested(itemPriority, /*commaIsOperator=*/false).term;
}

// Arguments or elements up to `close`, separated by ','. A list may end
// with '|' and its tail.
bool TermReader::ParseArguments(TokenKind close,
    llvm::SmallVectorImpl<const Term *> &arguments, const Term **tail) {
  while (true) {
    arguments.push_back(ParseArgument());
    if (failed_) {
      return false;
    }
    if (PeekKind() == TokenKind::Comma) {
      Next();
      continue;
    }
    if (tail && PeekKind() == TokenKind::HtSep) {
      Next();
      *tail = ParseArgument();
      if (failed_) {
        return false;
      }
      if (PeekKind() != close) {
        SayUnexpected("an operator or ']'");
        return false;
      }
    }
    if (PeekKind() == close) {
      Next();
      return true;
    }
    SayUnexpected(close == TokenKind::Close  ? "an operator, ',' or ')'"
            : close == TokenKind::CloseCurly ? "an operator, ',' or '}'"
                                             : "an operator, ',', '|' or ']'");
    return false;
  }
}

// A higher-order term: Closure(Args), possibly applied again.
const Term *TermReader::ParseApplications(
    const Term *term, std::uint32_t first) {
  while (!failed_ && PeekKind() == TokenKind::OpenCt) {
    Next();
    llvm::SmallVector<const Term *, 8> arguments;
    if (!ParseArguments(TokenKind::Close, arguments)) {
      return nullptr;
    }
    term = arena_.Make<ApplyTerm>(Range(first), term, arena_.Save(arguments));
  }
  return term;
}

// The operator between backquotes: a name, which may be an operator, a
// module-qualified name, or a variable.
const Term *TermReader::ParseBackquoteOperator() {
  const Term *op;
  if (PeekKind() == TokenKind::Name && PeekKind(1) == TokenKind::Backquote) {
    op = MakeName(Next());
  } else {
    // Only '.' (1490) binds this tightly, for module qualifiers.
    op = Parse(FindOperator(".")->infix->priority).term;
    if (failed_) {
      return nullptr;
    }
  }
  if (PeekKind() != TokenKind::Backquote) {
    SayUnexpected("'`'");
    return nullptr;
  }
  Next();
  return op;
}

// As in mmc, a float too large for a double is infinity.
const Term *TermReader::MakeLiteral(
    std::uint32_t first, std::uint32_t token, bool negative) {
  if (tokens_.tokens[token].kind == TokenKind::Integer) {
    return MakeInteger(first, token, negative);
  }
  std::string digits;
  for (char ch : tokens_.tokens[token].spelling) {
    if (ch != '_') {
      digits += ch;
    }
  }
  double value{std::strtod(digits.c_str(), nullptr)};
  // As in mmc, a negated zero is zero, not negative zero.
  return arena_.Make<FloatTerm>(
      Range(first), negative && value != 0 ? -value : value);
}

// The value of an integer literal, of any size; the lexer has checked its
// form.
const Term *TermReader::MakeInteger(
    std::uint32_t first, std::uint32_t token, bool negative) {
  llvm::StringRef spelling{tokens_.tokens[token].spelling};
  llvm::APInt magnitude{64, 0};
  bool isUnsigned{false};
  IntegerSize size{IntegerSize::Default};
  if (spelling.starts_with("0'")) {
    llvm::StringRef rest{spelling.drop_front(2)};
    const auto *p{reinterpret_cast<const llvm::UTF8 *>(rest.begin())};
    llvm::UTF32 code{0};
    if (!rest.empty() &&
        llvm::convertUTF8Sequence(&p,
            reinterpret_cast<const llvm::UTF8 *>(rest.end()), &code,
            llvm::strictConversion) == llvm::conversionOK) {
      magnitude = code;
    } else if (!rest.empty()) {
      magnitude = static_cast<unsigned char>(rest[0]);
    }
  } else {
    unsigned radix{10};
    if (spelling.size() > 2 && spelling[0] == '0') {
      switch (spelling[1]) {
      case 'b':
        radix = 2;
        break;
      case 'o':
        radix = 8;
        break;
      case 'x':
        radix = 16;
        break;
      }
    }
    std::size_t j{radix == 10 ? 0u : 2u};
    for (; j < spelling.size(); ++j) {
      char ch{spelling[j]};
      if (ch == '_') {
        continue;
      }
      unsigned digit;
      if (llvm::isDigit(ch)) {
        digit = ch - '0';
      } else if (radix == 16 && llvm::isHexDigit(ch)) {
        digit = llvm::hexDigitValue(ch);
      } else {
        break; // the suffix
      }
      if (magnitude.getActiveBits() + 5 > magnitude.getBitWidth()) {
        magnitude = magnitude.zext(magnitude.getBitWidth() + 64);
      }
      magnitude = magnitude * radix + digit;
    }
    llvm::StringRef suffix{spelling.substr(j)};
    if (!suffix.empty()) {
      isUnsigned = suffix[0] == 'u';
      llvm::StringRef bits{suffix.drop_front()};
      size = bits == "8" ? IntegerSize::Bits8
          : bits == "16" ? IntegerSize::Bits16
          : bits == "32" ? IntegerSize::Bits32
          : bits == "64" ? IntegerSize::Bits64
                         : IntegerSize::Default;
    }
  }
  unsigned words{std::max(1u, (magnitude.getActiveBits() + 63) / 64)};
  llvm::APInt trimmed{magnitude.zextOrTrunc(64 * words)};
  return arena_.Make<IntegerTerm>(Range(first),
      arena_.Save(llvm::ArrayRef<std::uint64_t>(trimmed.getRawData(), words)),
      negative && !magnitude.isZero(), isUnsigned, size);
}

const VariableTerm *TermReader::MakeVariable(std::uint32_t token) {
  llvm::StringRef name{tokens_.tokens[token].spelling};
  unsigned number;
  if (name == "_") {
    number = variables_++;
  } else {
    auto [iter, inserted]{variableNumbers_.try_emplace(name, variables_)};
    if (inserted) {
      ++variables_;
    }
    number = iter->second;
  }
  return arena_.Make<VariableTerm>(TokenRange{token, token}, name, number);
}

const NameTerm *TermReader::MakeName(std::uint32_t token) {
  llvm::StringRef spelling{tokens_.tokens[token].spelling};
  llvm::StringRef value{spelling};
  if (spelling.starts_with("'")) {
    value = arena_.Save(UnquoteSpelling(spelling));
  }
  return arena_.Make<NameTerm>(TokenRange{token, token}, value);
}

} // namespace merkurius::parser
