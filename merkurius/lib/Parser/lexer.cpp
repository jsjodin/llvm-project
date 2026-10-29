//===-- lib/Parser/lexer.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The rules are those of the Mercury 22.01 reference manual, section 2.4,
// "Tokens". Where it says nothing, the current manual's Syntax chapter
// fills the gap: comments, graphic names, ';' and the backquote. Its
// changes since 22.01 (the '\e' escape, underscores before an exponent,
// '<<u' and '>>u') are not adopted.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/source.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include <climits>

namespace merkurius::parser {

namespace {

bool IsWhitespace(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' ||
      ch == '\r';
}
bool IsBlank(char ch) { return ch == ' ' || ch == '\t'; }
bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }
bool IsBinaryDigit(char ch) { return ch == '0' || ch == '1'; }
bool IsOctalDigit(char ch) { return ch >= '0' && ch <= '7'; }
bool IsHexDigit(char ch) { return llvm::isHexDigit(ch); }
bool IsLower(char ch) { return ch >= 'a' && ch <= 'z'; }
bool IsUpper(char ch) { return ch >= 'A' && ch <= 'Z'; }
// Letters, digits and underscores, which continue names and variables.
bool IsAlphanumeric(char ch) {
  return IsLower(ch) || IsUpper(ch) || IsDigit(ch) || ch == '_';
}
bool IsGraphic(char ch) {
  return llvm::StringRef{"!&*+-:<=>?@^~\\#$./"}.contains(ch);
}

class Lexer {
public:
  // A lexer for the text [begin, end) of `file`. A directive's lexer has no
  // file: it sees one line of the directive, and finds neither directives
  // nor line number directives in it.
  Lexer(const char *begin, const char *end, const char *fileBegin,
      SourceFile *file, bool openMP, std::vector<LexerMessage> &messages)
      : at_{begin}, end_{end}, fileBegin_{fileBegin}, file_{file},
        openMP_{openMP}, messages_{messages} {}

  void LexFile(TokenSequence &);

private:
  bool AtEnd() const { return at_ == end_; }
  char Peek(std::size_t ahead = 0) const {
    return at_ + ahead < end_ ? at_[ahead] : '\0';
  }
  bool LookingAt(llvm::StringRef s) const {
    return llvm::StringRef(at_, end_ - at_).starts_with(s);
  }
  const char *LineEnd(const char *from) const {
    const char *p{from};
    while (p < end_ && *p != '\n') {
      ++p;
    }
    return p;
  }
  void Say(const char *begin, const char *end, std::string text) {
    messages_.push_back({CharBlock{begin, end}, std::move(text)});
  }
  void SkipCharacter() {
    unsigned n{llvm::getNumBytesForUTF8(static_cast<llvm::UTF8>(*at_))};
    at_ += std::min<std::size_t>(std::max(n, 1u), end_ - at_);
  }

  // Skips whitespace, comments and line number directives. Returns true
  // when it stops at a %$omp directive.
  bool SkipTrivia();
  bool AtDirective() const;
  bool SkipLineNumberDirective();
  TokenKind LexToken(bool precededByWhitespace);
  void LexDirective(Token &, std::vector<Token> &directiveTokens);
  TokenKind LexNumber();
  bool SkipDigits(bool (*isDigit)(char));
  void SkipIntegerSuffix(const char *start, bool trailingUnderscore);
  void SkipQuoted(char quote);
  void SkipEscape();
  void SkipHexDigits(const char *escape, int count);

  const char *at_;
  const char *const end_;
  const char *const fileBegin_;
  SourceFile *const file_;
  const bool openMP_;
  std::vector<LexerMessage> &messages_;
};

void Lexer::LexFile(TokenSequence &result) {
  const char *previousEnd{at_};
  while (true) {
    bool directive{SkipTrivia()};
    Token token;
    token.trivia = CharBlock{previousEnd, at_};
    const char *start{at_};
    if (directive) {
      token.kind = TokenKind::Directive;
      LexDirective(token, result.directiveTokens);
    } else if (AtEnd()) {
      token.kind = TokenKind::Eof;
    } else {
      token.kind = LexToken(!token.trivia.empty() || result.tokens.empty());
    }
    token.spelling = CharBlock{start, at_};
    result.tokens.push_back(token);
    if (token.kind == TokenKind::Eof) {
      return;
    }
    previousEnd = at_;
  }
}

bool Lexer::SkipTrivia() {
  while (!AtEnd()) {
    char ch{*at_};
    if (IsWhitespace(ch)) {
      ++at_;
    } else if (ch == '%') {
      if (AtDirective()) {
        return true;
      }
      at_ = LineEnd(at_);
    } else if (ch == '/' && Peek(1) == '*') {
      const char *start{at_};
      llvm::StringRef rest{at_ + 2, static_cast<std::size_t>(end_ - at_ - 2)};
      std::size_t close{rest.find("*/")};
      if (close == llvm::StringRef::npos) {
        at_ = end_;
        Say(start, start + 2, "unterminated comment");
      } else {
        at_ += 2 + close + 2;
      }
    } else if (!(ch == '#' && SkipLineNumberDirective())) {
      break;
    }
  }
  return false;
}

// A %$omp directive is a '%' comment that is the first non-blank text of
// its line and begins "%$omp" followed by whitespace.
bool Lexer::AtDirective() const {
  if (!openMP_ || !file_ || !LookingAt("%$omp") ||
      !(at_ + 5 == end_ || IsWhitespace(at_[5]))) {
    return false;
  }
  for (const char *p{at_}; p > fileBegin_ && p[-1] != '\n'; --p) {
    if (!IsBlank(p[-1])) {
      return false;
    }
  }
  return true;
}

// '#', a positive integer and a newline set the line number of the next
// line. A carriage return before the newline is accepted.
bool Lexer::SkipLineNumberDirective() {
  if (!file_) {
    return false;
  }
  const char *p{at_ + 1};
  while (p < end_ && IsDigit(*p)) {
    ++p;
  }
  llvm::StringRef digits{at_ + 1, static_cast<std::size_t>(p - at_ - 1)};
  if (digits.empty()) {
    return false;
  }
  const char *newline{p};
  if (newline < end_ && *newline == '\r') {
    ++newline;
  }
  if (newline == end_ || *newline != '\n') {
    return false;
  }
  const char *start{at_};
  at_ = newline + 1;
  unsigned long long line;
  if (digits.getAsInteger(10, line) || line == 0 || line > INT_MAX) {
    Say(start, p, "line number must be a positive integer");
    return true;
  }
  // A directive that ends the file has no line to number.
  std::size_t next{static_cast<std::size_t>(at_ - fileBegin_)};
  if (next < file_->bytes()) {
    SourcePosition here{
        file_->GetSourcePosition(static_cast<std::size_t>(start - fileBegin_))};
    file_->LineDirective(file_->GetSourcePosition(next).trueLineNumber,
        *here.path, static_cast<int>(line));
  }
  return true;
}

TokenKind Lexer::LexToken(bool precededByWhitespace) {
  const char *start{at_};
  char ch{*at_};
  if (IsDigit(ch)) {
    return LexNumber();
  }
  if (IsLower(ch) || IsUpper(ch) || ch == '_') {
    while (!AtEnd() && IsAlphanumeric(*at_)) {
      ++at_;
    }
    return IsLower(ch) ? TokenKind::Name : TokenKind::Variable;
  }
  if (ch == '$' && IsLower(Peek(1))) {
    ++at_;
    while (!AtEnd() && IsAlphanumeric(*at_)) {
      ++at_;
    }
    return TokenKind::ImplementationDefinedLiteral;
  }
  if (ch == '"' || ch == '\'') {
    SkipQuoted(ch);
    return ch == '"' ? TokenKind::String : TokenKind::Name;
  }
  if (IsGraphic(ch) && ch != '#') {
    // A graphic name stops before "/*", which begins a comment.
    while (!AtEnd() && IsGraphic(*at_) && !LookingAt("/*")) {
      ++at_;
    }
    if (at_ == start + 1 && ch == '.' &&
        (AtEnd() || IsWhitespace(*at_) || *at_ == '%' || LookingAt("/*"))) {
      return TokenKind::End;
    }
    return TokenKind::Name;
  }
  ++at_;
  switch (ch) {
  case '(':
    return precededByWhitespace ? TokenKind::Open : TokenKind::OpenCt;
  case ')':
    return TokenKind::Close;
  case '[':
    return TokenKind::OpenList;
  case ']':
    return TokenKind::CloseList;
  case '{':
    return TokenKind::OpenCurly;
  case '}':
    return TokenKind::CloseCurly;
  case '|':
    return TokenKind::HtSep;
  case ',':
    return TokenKind::Comma;
  case '`':
    return TokenKind::Backquote;
  case ';':
    return TokenKind::Name;
  }
  at_ = start;
  SkipCharacter();
  Say(start, at_, "unexpected character");
  return TokenKind::Error;
}

// A directive is its first line and each following line whose first
// non-blank text is "%$omp&". Its tokens are lexed line by line, after the
// sentinels; the sentinels and line breaks become trivia.
void Lexer::LexDirective(
    Token &directive, std::vector<Token> &directiveTokens) {
  struct Line {
    const char *begin, *end;
  };
  llvm::SmallVector<Line, 2> lines;
  const char *directiveEnd{LineEnd(at_)};
  lines.push_back({at_ + 5, directiveEnd});
  while (directiveEnd != end_) { // at a newline
    const char *p{directiveEnd + 1};
    while (p < end_ && IsBlank(*p)) {
      ++p;
    }
    if (!llvm::StringRef(p, end_ - p).starts_with("%$omp&")) {
      break;
    }
    directiveEnd = LineEnd(p);
    lines.push_back({p + 6, directiveEnd});
  }

  directive.firstDirectiveToken = directiveTokens.size();
  const char *previousEnd{at_};
  for (const Line &line : lines) {
    Lexer lexer{line.begin, line.end, fileBegin_, nullptr, false, messages_};
    while (true) {
      lexer.SkipTrivia();
      if (lexer.AtEnd()) {
        break;
      }
      Token token;
      token.trivia = CharBlock{previousEnd, lexer.at_};
      const char *start{lexer.at_};
      token.kind = lexer.LexToken(!token.trivia.empty());
      token.spelling = CharBlock{start, lexer.at_};
      directiveTokens.push_back(token);
      previousEnd = lexer.at_;
    }
  }
  Token eof;
  eof.kind = TokenKind::Eof;
  eof.trivia = CharBlock{previousEnd, directiveEnd};
  eof.spelling = CharBlock{directiveEnd, directiveEnd};
  directiveTokens.push_back(eof);
  directive.directiveTokens =
      directiveTokens.size() - directive.firstDirectiveToken;
  at_ = directiveEnd;
}

// Skips digits and underscores, starting at a digit. Returns true if the
// last character skipped was an underscore.
bool Lexer::SkipDigits(bool (*isDigit)(char)) {
  bool underscore{false};
  while (!AtEnd() && (isDigit(*at_) || *at_ == '_')) {
    underscore = *at_ == '_';
    ++at_;
  }
  return underscore;
}

// The suffixes i, i8, i16, i32, i64, u, u8, u16, u32 and u64. Underscores
// may come between the last digit and a suffix, but not end a literal.
void Lexer::SkipIntegerSuffix(const char *start, bool trailingUnderscore) {
  if (Peek() == 'i' || Peek() == 'u') {
    const char *p{at_ + 1};
    while (p < end_ && IsDigit(*p)) {
      ++p;
    }
    if (p == end_ || !IsAlphanumeric(*p)) {
      llvm::StringRef size{at_ + 1, static_cast<std::size_t>(p - at_ - 1)};
      const char *suffix{at_};
      at_ = p;
      if (!(size.empty() || size == "8" || size == "16" || size == "32" ||
              size == "64")) {
        Say(suffix, at_, "invalid integer suffix");
      }
      return;
    }
  }
  if (trailingUnderscore) {
    Say(start, at_, "an integer literal may not end with an underscore");
  }
}

TokenKind Lexer::LexNumber() {
  const char *start{at_};
  if (*at_ == '0') {
    char next{Peek(1)};
    if (next == '\'') { // a character code: 0' and any one character
      at_ += 2;
      if (AtEnd()) {
        Say(start, at_, "missing character after 0'");
      } else {
        SkipCharacter();
      }
      return TokenKind::Integer;
    }
    bool (*isDigit)(char){next == 'b' ? IsBinaryDigit
            : next == 'o'             ? IsOctalDigit
            : next == 'x'             ? IsHexDigit
                                      : nullptr};
    if (isDigit) {
      at_ += 2;
      while (!AtEnd() && *at_ == '_') {
        ++at_;
      }
      if (AtEnd() || !isDigit(*at_)) {
        Say(start, at_,
            "missing digits after '" + std::string(start, 2) + "'");
        return TokenKind::Integer;
      }
      bool trailingUnderscore{SkipDigits(isDigit)};
      if (!AtEnd() && IsDigit(*at_)) { // e.g. 0b102
        const char *digit{at_};
        SkipDigits(IsDigit);
        Say(digit, at_, "invalid digit in a literal with prefix '" +
                std::string(start, 2) + "'");
        return TokenKind::Integer;
      }
      SkipIntegerSuffix(start, trailingUnderscore);
      return TokenKind::Integer;
    }
  }
  bool trailingUnderscore{SkipDigits(IsDigit)};
  // A float has a fraction, an exponent, or both.
  bool fraction{Peek() == '.' && IsDigit(Peek(1))};
  auto atExponent{[&]() {
    return (Peek() == 'e' || Peek() == 'E') &&
        (IsDigit(Peek(1)) ||
            ((Peek(1) == '+' || Peek(1) == '-') && IsDigit(Peek(2))));
  }};
  if (!fraction && !atExponent()) {
    SkipIntegerSuffix(start, trailingUnderscore);
    return TokenKind::Integer;
  }
  auto sayUnderscore{[&](char next) {
    if (next == 'e' || next == 'E') {
      Say(start, at_,
          "an underscore before an exponent is not allowed in Mercury 22.01");
    } else if (next == '.') {
      Say(start, at_, "an underscore may not come before '.'");
    } else {
      Say(start, at_, "a float literal may not end with an underscore");
    }
  }};
  if (trailingUnderscore) {
    sayUnderscore(*at_);
  }
  if (fraction) {
    ++at_;
    if (SkipDigits(IsDigit)) {
      sayUnderscore(Peek());
    }
  }
  if (atExponent()) {
    at_ += IsDigit(Peek(1)) ? 1 : 2;
    if (SkipDigits(IsDigit)) {
      sayUnderscore('\0');
    }
  }
  return TokenKind::Float;
}

// A string or quoted name. A doubled quote stands for itself.
void Lexer::SkipQuoted(char quote) {
  const char *start{at_++};
  while (!AtEnd()) {
    if (*at_ == quote) {
      ++at_;
      if (Peek() != quote) {
        return;
      }
      ++at_;
    } else if (*at_ == '\\') {
      SkipEscape();
    } else {
      ++at_;
    }
  }
  Say(start, start + 1,
      quote == '"' ? "unterminated string" : "unterminated quoted name");
}

void Lexer::SkipHexDigits(const char *escape, int count) {
  for (int j{0}; j < count; ++j, ++at_) {
    if (AtEnd() || !IsHexDigit(*at_)) {
      Say(escape, at_,
          "expected " + std::to_string(count) + " hexadecimal digits");
      return;
    }
  }
}

void Lexer::SkipEscape() {
  const char *escape{at_++};
  if (AtEnd()) {
    return; // the caller reports the unterminated literal
  }
  char ch{*at_};
  switch (ch) {
  case 'a':
  case 'b':
  case 'r':
  case 'f':
  case 't':
  case 'n':
  case 'v':
  case '\\':
  case '\'':
  case '"':
  case '\n': // deleted, continuing the literal on the next line
    ++at_;
    return;
  case '\r':
    if (Peek(1) == '\n') {
      at_ += 2;
      return;
    }
    break;
  case 'e':
    ++at_;
    Say(escape, at_, "'\\e' is not an escape in Mercury 22.01");
    return;
  case 'u':
  case 'U': {
    ++at_;
    const char *digits{at_};
    int count{ch == 'u' ? 4 : 8};
    SkipHexDigits(escape, count);
    unsigned code;
    if (at_ - digits == count &&
        !llvm::StringRef(digits, count).getAsInteger(16, code) &&
        code > 0x10FFFF) {
      Say(escape, at_, "a Unicode escape may not exceed \\U0010FFFF");
    }
    return;
  }
  }
  if (ch == 'x' || IsOctalDigit(ch)) {
    bool hex{ch == 'x'};
    if (hex) {
      ++at_;
    }
    const char *digits{at_};
    while (!AtEnd() && (hex ? IsHexDigit(*at_) : IsOctalDigit(*at_))) {
      ++at_;
    }
    if (at_ == digits) {
      Say(escape, at_, "expected hexadecimal digits after '\\x'");
    } else if (Peek() != '\\') {
      Say(escape, at_,
          std::string{"missing '\\' to end the "} +
              (hex ? "hexadecimal" : "octal") + " escape");
    } else {
      ++at_;
    }
    return;
  }
  SkipCharacter();
  Say(escape, at_, "invalid escape sequence");
}

} // namespace

TokenSequence Lex(SourceFile &file, const LexerOptions &options,
    std::vector<LexerMessage> &messages) {
  TokenSequence result;
  llvm::StringRef text{file.content()};
  Lexer lexer{
      text.begin(), text.end(), text.begin(), &file, options.openMP, messages};
  lexer.LexFile(result);
  return result;
}

} // namespace merkurius::parser
