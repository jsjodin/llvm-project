//===-- lib/Parser/unparse-canonical.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The canonical unparser. It prints the tokens of the parse tree in order,
// with their spellings and comments, and chooses only the layout between
// them, from the tree: one goal per line in a clause body, if-then-else and
// disjunction as in the reference manual, constructors under '--->', and
// line breaking by a Wadler-style pretty-printer. Guards keep the layout from
// changing how the text lexes: nothing goes before an open_ct or after a '.'
// name, an open keeps the layout before it, two tokens that would run
// together are kept apart, and within a name, variable or literal the
// layout is the source's. Items that were left out for an error are printed
// as written.

#include "merkurius/Parser/unparse.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace merkurius::parser {

namespace {

//===----------------------------------------------------------------------===//
// The pretty-printer
//===----------------------------------------------------------------------===//

// The ways to separate two tokens, weakest first. A SoftLine is nothing and
// a Line a space when their group fits on the line, and both are a line
// break when it does not.
enum class Break : std::uint8_t {
  None,
  Space,
  SoftLine,
  Line,
  HardLine,
  BlankLine,
};

struct Command {
  enum class Kind : std::uint8_t {
    Text,
    Separator,
    BeginGroup,
    EndGroup,
    BeginNest,
    EndNest,
  };
  Kind kind;
  llvm::StringRef text; // for Text
  Break brk{Break::None}; // for a Separator
  unsigned count{1}; // spaces for Break::Space, indentation for BeginNest
  // For a Separator: comments printed before the break, each after a space,
  // and comments on lines of their own after it, where an empty one is a
  // blank line.
  llvm::SmallVector<llvm::StringRef, 2> trailing;
  llvm::SmallVector<llvm::StringRef, 2> leading;
};

constexpr std::uint64_t infinite{std::uint64_t{1} << 40};

bool IsLineComment(llvm::StringRef comment) {
  return comment.starts_with("%");
}

std::uint64_t TextWidth(llvm::StringRef text) {
  return text.contains('\n') ? infinite : text.size();
}

// The width of a separator in a group that fits, or `infinite` if it must
// break the line.
std::uint64_t FlatWidth(const Command &c) {
  std::uint64_t width{0};
  for (llvm::StringRef comment : c.trailing) {
    if (IsLineComment(comment)) {
      return infinite;
    }
    width += 1 + TextWidth(comment);
  }
  if (!c.leading.empty()) {
    return infinite;
  }
  switch (c.brk) {
  case Break::None:
  case Break::SoftLine:
    return width;
  case Break::Space:
    return width + c.count;
  case Break::Line:
    return width + 1;
  default:
    return infinite;
  }
}

class Document {
public:
  std::size_t Add(Command::Kind kind, llvm::StringRef text = {},
      Break brk = Break::None, unsigned count = 1) {
    commands_.push_back(Command{kind, text, brk, count, {}, {}});
    return commands_.size() - 1;
  }
  void Text(llvm::StringRef text) { Add(Command::Kind::Text, text); }
  std::size_t Separator(Break brk, unsigned count = 1) {
    return Add(Command::Kind::Separator, {}, brk, count);
  }
  void BeginGroup() { Add(Command::Kind::BeginGroup); }
  void EndGroup() { Add(Command::Kind::EndGroup); }
  void BeginNest(unsigned indent) {
    Add(Command::Kind::BeginNest, {}, Break::None, indent);
  }
  void EndNest() { Add(Command::Kind::EndNest); }
  Command &operator[](std::size_t j) { return commands_[j]; }

  void Render(llvm::raw_ostream &o, unsigned width) const;

private:
  std::vector<Command> commands_;
};

// A group is printed flat if it fits on the line together with the text
// after it up to the next place the line may break.
void Document::Render(llvm::raw_ostream &o, unsigned width) const {
  std::size_t n{commands_.size()};
  // The width of the text from each command to the next possible break.
  std::vector<std::uint64_t> rest(n + 1, 0);
  for (std::size_t j{n}; j-- > 0;) {
    const Command &c{commands_[j]};
    if (c.kind == Command::Kind::Text) {
      std::size_t newline{c.text.find('\n')};
      rest[j] = newline != llvm::StringRef::npos ? newline
                                                 : c.text.size() + rest[j + 1];
    } else if (c.kind == Command::Kind::Separator) {
      std::uint64_t flat{FlatWidth(c)};
      rest[j] = c.brk >= Break::SoftLine || flat >= infinite
          ? 0
          : flat + rest[j + 1];
    } else {
      rest[j] = rest[j + 1];
    }
  }
  // The width of the commands before each one, printed flat, and how many
  // of them cannot be, from which a group's flat width follows.
  std::vector<std::uint64_t> before(n + 1, 0);
  std::vector<std::size_t> breaking(n + 1, 0);
  for (std::size_t j{0}; j < n; ++j) {
    const Command &c{commands_[j]};
    std::uint64_t width{c.kind == Command::Kind::Text ? TextWidth(c.text)
            : c.kind == Command::Kind::Separator      ? FlatWidth(c)
                                                      : 0};
    before[j + 1] = before[j] + (width >= infinite ? 0 : width);
    breaking[j + 1] = breaking[j] + (width >= infinite ? 1 : 0);
  }
  std::vector<std::size_t> groupEnd(n, n);
  std::vector<std::size_t> open;
  for (std::size_t j{0}; j < n; ++j) {
    if (commands_[j].kind == Command::Kind::BeginGroup) {
      open.push_back(j);
    } else if (commands_[j].kind == Command::Kind::EndGroup && !open.empty()) {
      groupEnd[open.back()] = j;
      open.pop_back();
    }
  }

  std::uint64_t column{0};
  bool lineStart{true};
  std::vector<unsigned> indents{0};
  std::vector<bool> flat{false};
  // A line is indented as the nest where it began, and written lazily, so
  // that blank lines have no trailing spaces.
  unsigned lineIndent{0};
  auto newline{[&]() {
    o << '\n';
    column = 0;
    lineStart = true;
    lineIndent = indents.back();
  }};
  auto text{[&](llvm::StringRef t) {
    if (t.empty()) {
      return;
    }
    if (lineStart) {
      o.indent(lineIndent);
      column = lineIndent;
      lineStart = false;
    }
    o << t;
    std::size_t newline{t.rfind('\n')};
    column = newline == llvm::StringRef::npos ? column + t.size()
                                              : t.size() - newline - 1;
  }};
  for (std::size_t j{0}; j < n; ++j) {
    const Command &c{commands_[j]};
    switch (c.kind) {
    case Command::Kind::Text:
      text(c.text);
      break;
    case Command::Kind::Separator: {
      Break brk{c.brk};
      for (llvm::StringRef comment : c.trailing) {
        if (!lineStart) {
          o << ' ';
          ++column;
        }
        text(comment);
        if (IsLineComment(comment)) {
          brk = std::max(brk, Break::HardLine);
        }
      }
      if (!c.leading.empty() && !lineStart) {
        brk = std::max(brk, Break::HardLine);
      }
      switch (brk) {
      case Break::None:
        break;
      case Break::Space:
        if (!lineStart) {
          o.indent(c.count);
          column += c.count;
        }
        break;
      case Break::SoftLine:
      case Break::Line:
        if (!flat.back()) {
          newline();
        } else if (brk == Break::Line) {
          o << ' ';
          ++column;
        }
        break;
      case Break::HardLine:
        newline();
        break;
      case Break::BlankLine:
        newline();
        newline();
        break;
      }
      for (llvm::StringRef comment : c.leading) {
        if (comment.empty()) {
          newline();
        } else {
          text(comment);
          newline();
        }
      }
      break;
    }
    case Command::Kind::BeginGroup: {
      std::uint64_t at{lineStart ? lineIndent : column};
      std::size_t end{groupEnd[j]};
      std::uint64_t needed{
          at + (before[end] - before[j]) + (end < n ? rest[end + 1] : 0)};
      flat.push_back(flat.back() ||
          (breaking[end] == breaking[j] && needed <= width));
      break;
    }
    case Command::Kind::EndGroup:
      if (flat.size() > 1) {
        flat.pop_back();
      }
      break;
    case Command::Kind::BeginNest:
      indents.push_back(indents.back() + c.count);
      break;
    case Command::Kind::EndNest:
      if (indents.size() > 1) {
        indents.pop_back();
      }
      break;
    }
  }
  if (!lineStart) {
    o << '\n';
  }
}

//===----------------------------------------------------------------------===//
// Trivia
//===----------------------------------------------------------------------===//

// The comments in a token's trivia: those on the line of the token before
// it, which follow that token, and those on lines of their own, which
// precede this one. Line number directives are dropped.
struct Comments {
  llvm::SmallVector<llvm::StringRef, 2> trailing;
  // With blank lines kept, an empty entry for each blank line before a
  // comment or before the token.
  llvm::SmallVector<llvm::StringRef, 2> leading;
};

bool IsWhitespace(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' ||
      ch == '\v';
}

// `onNewLine` treats the trivia as starting on a line of its own, as at the
// start of the file, where no blank line is kept before the first comment.
Comments ParseTrivia(
    llvm::StringRef trivia, bool keepBlankLines, bool onNewLine) {
  Comments comments;
  bool seenNewline{onNewLine};
  unsigned newlines{onNewLine ? 1u : 0u};
  bool first{true};
  auto comment{[&](llvm::StringRef text) {
    text = text.rtrim("\r");
    if (!seenNewline) {
      comments.trailing.push_back(text);
    } else {
      if (keepBlankLines && newlines >= 2 && !(onNewLine && first)) {
        comments.leading.push_back({});
      }
      comments.leading.push_back(text);
      first = false;
    }
    newlines = 0;
  }};
  std::size_t j{0};
  while (j < trivia.size()) {
    char ch{trivia[j]};
    if (ch == '\n') {
      ++newlines;
      seenNewline = true;
      ++j;
    } else if (IsWhitespace(ch)) {
      ++j;
    } else if (ch == '%') {
      std::size_t end{trivia.find('\n', j)};
      end = end == llvm::StringRef::npos ? trivia.size() : end;
      comment(trivia.slice(j, end));
      j = end;
    } else if (ch == '/' && j + 1 < trivia.size() && trivia[j + 1] == '*') {
      std::size_t close{trivia.find("*/", j + 2)};
      std::size_t end{close == llvm::StringRef::npos ? trivia.size()
                                                     : close + 2};
      comment(trivia.slice(j, end));
      j = end;
    } else if (ch == '#') {
      // A line number directive, which ends with its newline.
      std::size_t end{trivia.find('\n', j)};
      j = end == llvm::StringRef::npos ? trivia.size() : end;
    } else {
      ++j; // a byte order mark
    }
  }
  if (keepBlankLines && newlines >= 2 &&
      !(onNewLine && comments.leading.empty())) {
    comments.leading.push_back({});
  }
  return comments;
}

bool IsGraphicChar(char ch) {
  return llvm::StringRef{"!&*+-:<=>?@^~\\#$./"}.contains(ch);
}

bool IsAlphanumericChar(char ch) {
  return llvm::isAlnum(ch) || ch == '_' || static_cast<unsigned char>(ch) >= 0x80;
}

// Whether two spellings would run together into other tokens if nothing
// separated them.
bool RunTogether(llvm::StringRef left, llvm::StringRef right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  char a{left.back()}, b{right.front()};
  return (IsGraphicChar(a) && IsGraphicChar(b)) ||
      (IsAlphanumericChar(a) && IsAlphanumericChar(b)) ||
      (a == b && (a == '"' || a == '\'' || a == '`')) ||
      (llvm::isDigit(a) && b == '\'');
}

bool IsOpenKind(TokenKind kind) {
  return kind == TokenKind::Open || kind == TokenKind::OpenCt ||
      kind == TokenKind::OpenList || kind == TokenKind::OpenCurly;
}

bool IsCloseKind(TokenKind kind) {
  return kind == TokenKind::Close || kind == TokenKind::CloseList ||
      kind == TokenKind::CloseCurly;
}

//===----------------------------------------------------------------------===//
// The canonical unparser
//===----------------------------------------------------------------------===//

template <typename T> const void *Pointer(const T *p) { return p; }
template <typename... A> const void *Pointer(const std::variant<A...> &v) {
  return std::visit([](const auto *p) -> const void * { return p; }, v);
}
template <typename T> const void *Pointer(const std::optional<T> &o) {
  return o ? Pointer(*o) : nullptr;
}

// Nodes whose tokens are printed with the source's layout between them.
template <typename N>
constexpr bool isAtomic{std::is_same_v<N, SymName> ||
    std::is_same_v<N, Variable> || std::is_same_v<N, StateVar> ||
    std::is_same_v<N, Literal> || std::is_same_v<N, ImplementationLiteral> ||
    std::is_same_v<N, TypeVar>};

template <typename N, typename... A>
constexpr bool isOneOf{(std::is_same_v<N, A> || ...)};

template <typename N>
constexpr bool isDisjunction{isOneOf<N, Disjunction, DcgDisjunction>};
template <typename N>
constexpr bool isIfThenElse{isOneOf<N, IfThenElse, DcgIfThenElse>};
template <typename N>
constexpr bool isConjunction{isOneOf<N, Conjunction, DcgConjunction>};

class CanonicalUnparser {
public:
  CanonicalUnparser(const TokenSequence &tokens, bool parenthesize)
      : tokens_{tokens.tokens}, parenthesize_{parenthesize},
        match_(tokens.tokens.size(), noMatch) {
    std::vector<std::uint32_t> open;
    for (std::uint32_t j{0}; j < tokens_.size(); ++j) {
      TokenKind kind{tokens_[j].kind};
      if (IsOpenKind(kind)) {
        open.push_back(j);
      } else if (IsCloseKind(kind) && !open.empty()) {
        TokenKind opener{tokens_[open.back()].kind};
        if ((kind == TokenKind::Close &&
                (opener == TokenKind::Open || opener == TokenKind::OpenCt)) ||
            (kind == TokenKind::CloseList && opener == TokenKind::OpenList) ||
            (kind == TokenKind::CloseCurly && opener == TokenKind::OpenCurly)) {
          match_[open.back()] = j;
          match_[j] = open.back();
        }
        open.pop_back();
      }
    }
  }

  void Unparse(const ParseTree &);
  void Render(llvm::raw_ostream &o) const { doc_.Render(o, 79); }

private:
  static constexpr std::uint32_t noMatch{~std::uint32_t{0}};

  struct Child {
    TokenRange tokens;
    llvm::StringRef kind;
    const void *node;
    void (CanonicalUnparser::*unparse)(const void *, bool, bool);
    bool viaVariant; // a data-term, goal, type, inst or mode
  };
  // A node's tokens and children, in order.
  struct Element {
    const Child *child; // or null for a token
    std::uint32_t token;
  };

  enum EmitFlags : unsigned {
    keepAdjacency = 1, // the source's layout before this token
    keepAdjacencyAfter = 2, // and after it
    tight = 4, // nothing before or after
    tightAfter = 8,
  };

  // The token printed last, which decides what may follow it.
  struct Last {
    bool valid{false};
    TokenKind kind{TokenKind::Eof};
    llvm::StringRef spelling;
    bool keepAdjacencyAfter{false};
    bool tightAfter{false};
  };

  template <typename T>
  void Gather(const T &value, llvm::SmallVectorImpl<Child> &children,
      bool viaVariant) {
    if constexpr (std::is_same_v<T, OmpDirective>) {
      // Its nodes index the directive's own tokens; the directive token is
      // printed as written.
    } else if constexpr (IsNode<T>::value) {
      children.push_back({value.tokens, T::kindName, &value,
          &CanonicalUnparser::UnparseNode<T>, viaVariant});
    } else if constexpr (std::is_pointer_v<T>) {
      if (value) {
        Gather(*value, children, viaVariant);
      }
    } else if constexpr (IsOptional<T>::value) {
      if (value) {
        Gather(*value, children, viaVariant);
      }
    } else if constexpr (IsVariant<T>::value) {
      std::visit([&](const auto &x) { Gather(x, children, true); }, value);
    } else if constexpr (IsArrayRef<T>::value) {
      for (const auto &element : value) {
        Gather(element, children, viaVariant);
      }
    }
  }

  template <typename N>
  void UnparseNode(const void *node, bool block, bool viaVariant) {
    Node(*static_cast<const N *>(node), block, viaVariant);
  }

  template <typename N> void Node(const N &, bool block, bool viaVariant);
  template <typename N>
  void Gap(const N &, llvm::ArrayRef<Element>, std::size_t e, bool block,
      unsigned &nests, unsigned &brackets, unsigned &backquotes);
  template <typename N> bool BeforeChild(const N &, const Child &);
  template <typename N> bool IsBlockChild(const N &, const Child &) const;
  template <typename N>
  bool IsOperatorTerm(const N &, llvm::ArrayRef<Child>) const;

  void Emit(std::uint32_t token, unsigned flags = 0);
  void EmitSynthetic(llvm::StringRef text, TokenKind kind);
  void Place(TokenKind kind, llvm::StringRef spelling, bool synthetic,
      bool sourceAdjacent, unsigned flags, Comments comments);
  void Request(Break brk, unsigned count = 1) {
    pending_.push_back(doc_.Separator(brk, count));
  }
  void StartItem(std::uint32_t token);
  void PrintAsWritten(std::uint32_t first, std::uint32_t last);

  const std::vector<Token> &tokens_;
  bool parenthesize_;
  std::vector<std::uint32_t> match_; // each bracket's partner
  Document doc_;
  Last last_;
  // The glue separator after the last token, and those requested since.
  std::size_t glue_{0};
  llvm::SmallVector<std::size_t, 4> pending_;
  bool itemStart_{false}; // the next token begins an item
  bool fileStart_{true};
};

void CanonicalUnparser::Emit(std::uint32_t j, unsigned flags) {
  const Token &token{tokens_[j]};
  Comments comments;
  if (!itemStart_) {
    comments = ParseTrivia(token.trivia, false, false);
  }
  Place(token.kind, token.spelling, false, token.trivia.empty(), flags,
      std::move(comments));
}

void CanonicalUnparser::EmitSynthetic(llvm::StringRef text, TokenKind kind) {
  Place(kind, text, true, false, 0, {});
}

// Resolves the separators since the last token into one, adds the comments
// to it, and appends the token.
void CanonicalUnparser::Place(TokenKind kind, llvm::StringRef spelling,
    bool synthetic, bool sourceAdjacent, unsigned flags, Comments comments) {
  if (itemStart_ || !last_.valid) {
    itemStart_ = false;
    for (std::size_t j : pending_) {
      doc_[j].brk = Break::None;
    }
  } else {
    enum class Need { Any, None, Space } need{Need::Any};
    Break glue{Break::Space};
    bool keepSource{(flags & keepAdjacency) || last_.keepAdjacencyAfter};
    if (kind == TokenKind::OpenCt &&
        !(IsOpenKind(last_.kind) || last_.kind == TokenKind::Comma ||
            last_.kind == TokenKind::HtSep)) {
      // After a term, layout before '(' would make an open of the open_ct
      // that applies the term. Elsewhere the two read the same.
      need = Need::None;
    } else if (kind == TokenKind::OpenCt) {
      glue = Break::None;
    } else if (last_.kind == TokenKind::Name && last_.spelling == ".") {
      // A '.' name followed by layout would be an end token.
      need = Need::None;
    } else if (keepSource && !synthetic) {
      if (sourceAdjacent) {
        need = Need::None;
      }
    } else if (IsOpenKind(last_.kind) || IsCloseKind(kind) ||
        kind == TokenKind::Comma || kind == TokenKind::End ||
        (flags & tight) || last_.tightAfter) {
      glue = Break::None;
    }
    if (need == Need::Any && kind == TokenKind::Open &&
        (!synthetic ||
            !(IsOpenKind(last_.kind) || last_.kind == TokenKind::Comma ||
                last_.kind == TokenKind::HtSep))) {
      // An open is a '(' after layout, unlike an open_ct.
      need = Need::Space;
    } else if (need == Need::Any && kind == TokenKind::Open) {
      glue = Break::None;
    }
    if (need == Need::Any && glue == Break::None &&
        RunTogether(last_.spelling, spelling)) {
      need = Need::Space;
    }

    // A directive is a line of its own.
    if (kind == TokenKind::Directive || last_.kind == TokenKind::Directive) {
      need = Need::Any;
      glue = Break::HardLine;
    }

    Break requested{Break::None};
    unsigned spaces{1};
    std::size_t chosen{glue_};
    for (std::size_t j : pending_) {
      if (doc_[j].brk > requested ||
          (doc_[j].brk == requested && requested != Break::None)) {
        if (doc_[j].brk == Break::Space && requested == Break::Space) {
          spaces = std::max(spaces, doc_[j].count);
        } else {
          spaces = doc_[j].count;
        }
        requested = doc_[j].brk;
        chosen = j;
      }
    }
    Break brk;
    if (need == Need::None) {
      brk = Break::None;
    } else {
      brk = std::max(requested, glue);
      if (requested < glue) {
        chosen = glue_;
        spaces = 1;
      }
      if (need == Need::Space && brk == Break::None) {
        brk = Break::Space;
        spaces = 1;
      } else if (need == Need::Space && brk == Break::SoftLine) {
        brk = Break::Line;
      }
    }
    doc_[glue_].brk = Break::None;
    for (std::size_t j : pending_) {
      doc_[j].brk = Break::None;
    }
    Command &separator{doc_[chosen]};
    separator.brk = brk;
    separator.count = spaces;
    separator.trailing = comments.trailing;
    separator.leading = comments.leading;
  }
  pending_.clear();
  if (kind == TokenKind::Directive) {
    // Its continuation lines at its own indentation.
    llvm::SmallVector<llvm::StringRef, 2> lines;
    spelling.split(lines, '\n');
    for (std::size_t j{0}; j < lines.size(); ++j) {
      if (j > 0) {
        doc_.Separator(Break::HardLine);
      }
      doc_.Text(lines[j].trim(" \t\r"));
    }
  } else {
    doc_.Text(spelling);
  }
  last_ = Last{true, kind, spelling,
      !synthetic && (flags & keepAdjacencyAfter) != 0,
      (flags & (tight | tightAfter)) != 0};
  glue_ = doc_.Separator(Break::None);
}

// Begins an item at `token`: the comments before it, with blank lines kept,
// after a line break.
void CanonicalUnparser::StartItem(std::uint32_t token) {
  llvm::StringRef trivia{tokens_[token].trivia};
  if (fileStart_ && trivia.starts_with("\xef\xbb\xbf")) {
    doc_.Text(trivia.take_front(3));
  }
  Comments comments{ParseTrivia(trivia, true, fileStart_)};
  std::size_t separator{
      doc_.Separator(fileStart_ ? Break::None : Break::HardLine)};
  doc_[separator].trailing = comments.trailing;
  doc_[separator].leading = comments.leading;
  for (std::size_t j : pending_) {
    doc_[j].brk = Break::None;
  }
  pending_.clear();
  fileStart_ = false;
  itemStart_ = true;
}

// Prints the items in [first, last), which were left out for errors, as
// written.
void CanonicalUnparser::PrintAsWritten(
    std::uint32_t first, std::uint32_t last) {
  std::uint32_t j{first};
  while (j < last) {
    std::uint32_t end{j};
    while (end + 1 < last && tokens_[end].kind != TokenKind::End) {
      ++end;
    }
    StartItem(j);
    itemStart_ = false;
    doc_.Text(tokens_[j].spelling);
    for (std::uint32_t k{j + 1}; k <= end; ++k) {
      doc_.Text(tokens_[k].trivia);
      doc_.Text(tokens_[k].spelling);
    }
    last_ = Last{true, tokens_[end].kind, tokens_[end].spelling};
    glue_ = doc_.Separator(Break::None);
    j = end + 1;
  }
}

void CanonicalUnparser::Unparse(const ParseTree &tree) {
  llvm::SmallVector<Child, 32> items;
  for (const Item &item : tree.items) {
    Gather(item, items, false);
  }
  auto eof{static_cast<std::uint32_t>(tokens_.size() - 1)};
  std::uint32_t cursor{0};
  for (const Child &item : items) {
    if (item.tokens.first < cursor || item.tokens.last >= eof) {
      continue;
    }
    std::uint32_t start{item.tokens.first};
    if (start > cursor && tokens_[start - 1].kind == TokenKind::Name &&
        tokens_[start - 1].spelling == ":-") {
      --start;
    }
    PrintAsWritten(cursor, start);
    StartItem(start);
    if (start != item.tokens.first) {
      Emit(start);
    }
    (this->*item.unparse)(item.node, false, false);
    cursor = item.tokens.last + 1;
    if (tokens_[cursor].kind == TokenKind::End) {
      Emit(cursor);
      ++cursor;
    }
  }
  PrintAsWritten(cursor, eof);
  // The comments at the end of the file.
  Comments comments{ParseTrivia(tokens_[eof].trivia, true, fileStart_)};
  std::size_t separator{doc_.Separator(Break::None)};
  doc_[separator].trailing = comments.trailing;
  doc_[separator].leading = comments.leading;
  for (std::size_t j : pending_) {
    doc_[j].brk = Break::None;
  }
}

template <typename N>
bool CanonicalUnparser::IsBlockChild(const N &n, const Child &c) const {
  if constexpr (isOneOf<N, FactOrRule, DcgRule>) {
    return c.node == Pointer(n.body);
  } else if constexpr (isIfThenElse<N>) {
    return c.node == Pointer(n.then) || c.node == Pointer(n.otherwise);
  } else if constexpr (isDisjunction<N>) {
    return true;
  } else if constexpr (isOneOf<N, QuantifiedGoal, DcgQuantified,
                           PromisePurity, PromiseSolutions,
                           RequireDeterminism, RequireSwitch, DisableWarnings,
                           TraceGoal, CatchClause>) {
    return c.node == Pointer(n.goal);
  } else if constexpr (std::is_same_v<N, TryGoal>) {
    return c.node == Pointer(n.goal) || c.node == Pointer(n.then) ||
        c.node == Pointer(n.otherwise);
  } else {
    return false;
  }
}

// Starts the layout of a child, and returns whether it began a nest for
// it.
template <typename N>
bool CanonicalUnparser::BeforeChild(const N &n, const Child &c) {
  auto nest{[&](Break brk) {
    doc_.BeginNest(4);
    Request(brk);
    return true;
  }};
  if constexpr (isOneOf<N, FactOrRule, DcgRule>) {
    if (c.node == Pointer(n.body)) {
      return nest(Break::HardLine);
    }
  } else if constexpr (std::is_same_v<N, Lambda>) {
    if (c.node == Pointer(n.body) || c.node == Pointer(n.dcgBody)) {
      return nest(Break::Line);
    }
  } else if constexpr (isIfThenElse<N>) {
    if (c.node == Pointer(n.then)) {
      return nest(Break::HardLine);
    }
    if (c.node == Pointer(n.otherwise)) {
      // else if C then ..., and ; C -> ..., continue the chain.
      llvm::StringRef keyword{n.form == IfThenElseForm::IfThenElse ? "else" : ";"};
      if (c.kind == N::kindName &&
          static_cast<const N *>(c.node)->form == n.form &&
          c.tokens.first > 0 &&
          tokens_[c.tokens.first - 1].spelling == keyword) {
        Request(Break::Space);
        return false;
      }
      return nest(Break::HardLine);
    }
  } else if constexpr (isDisjunction<N>) {
    return nest(Break::HardLine);
  }
  return false;
}

// Whether the node is written as an operator term: not a name, a variable,
// a literal, f(Args), P(Args), [...], {...} or (...).
template <typename N>
bool CanonicalUnparser::IsOperatorTerm(
    const N &n, llvm::ArrayRef<Child> children) const {
  std::uint32_t first{n.tokens.first}, last{n.tokens.last};
  if (first >= last || tokens_[first].kind == TokenKind::Directive) {
    return false;
  }
  std::uint32_t head{first};
  if (!children.empty() && children[0].kind == SymName::kindName &&
      children[0].tokens.first == first) {
    head = children[0].tokens.last;
  }
  if (head == last ||
      (tokens_[head + 1].kind == TokenKind::OpenCt &&
          match_[head + 1] == last)) {
    return false;
  }
  return !(IsOpenKind(tokens_[first].kind) && match_[first] == last);
}

template <typename N>
void CanonicalUnparser::Node(const N &n, bool block, bool viaVariant) {
  llvm::SmallVector<Child, 8> children;
  n.ForEachField(
      [&](llvm::StringRef, const auto &value) { Gather(value, children, false); });
  std::stable_sort(children.begin(), children.end(),
      [](const Child &x, const Child &y) {
        return x.tokens.first < y.tokens.first;
      });
  llvm::SmallVector<Element, 16> elements;
  std::uint32_t next{n.tokens.first};
  for (const Child &child : children) {
    if (child.tokens.first < next || child.tokens.first > child.tokens.last ||
        child.tokens.last > n.tokens.last) {
      continue; // the exact unparser reports these
    }
    for (std::uint32_t j{next}; j < child.tokens.first; ++j) {
      elements.push_back({nullptr, j});
    }
    elements.push_back({&child, 0});
    next = child.tokens.last + 1;
  }
  for (std::uint32_t j{next}; j <= n.tokens.last; ++j) {
    elements.push_back({nullptr, j});
  }

  bool wrap{parenthesize_ && viaVariant && IsOperatorTerm(n, children)};
  if (wrap) {
    EmitSynthetic("(", TokenKind::Open);
  }
  doc_.BeginGroup();
  if constexpr (isAtomic<N>) {
    for (std::uint32_t j{n.tokens.first}; j <= n.tokens.last; ++j) {
      Emit(j, j == n.tokens.first ? 0 : keepAdjacency);
    }
  } else {
    unsigned nests{0}, brackets{0}, backquotes{0};
    for (std::size_t e{0}; e < elements.size(); ++e) {
      if (const Child * c{elements[e].child}) {
        bool nested{BeforeChild(n, *c)};
        (this->*c->unparse)(c->node, IsBlockChild(n, *c), c->viaVariant);
        if (nested) {
          doc_.EndNest();
        }
        // A graphic prefix operator, as in -X, keeps the source's layout
        // after it.
        if (e == 0 && c->kind == SymName::kindName && elements.size() > 1 &&
            elements[1].child && IsGraphicChar(last_.spelling.front())) {
          last_.keepAdjacencyAfter = true;
        }
        continue;
      }
      // ( Goal ), with the goal on lines of its own: a disjunction or an
      // if-then-else, or a goal that is laid out as a block here.
      std::uint32_t j{elements[e].token};
      TokenKind kind{tokens_[j].kind};
      if ((kind == TokenKind::Open || kind == TokenKind::OpenCt) &&
          e + 2 < elements.size() && elements[e + 1].child &&
          !elements[e + 2].child && elements[e + 2].token == match_[j]) {
        const Child &c{*elements[e + 1].child};
        bool ite{c.kind == IfThenElse::kindName ||
            c.kind == DcgIfThenElse::kindName};
        bool disjunction{c.kind == Disjunction::kindName ||
            c.kind == DcgDisjunction::kindName};
        if (c.tokens.first == j + 1 && c.tokens.last + 1 == match_[j] &&
            (ite || disjunction || IsBlockChild(n, c))) {
          // The child's own layout, such as a clause body's line break,
          // comes before the parenthesis.
          bool nested{BeforeChild(n, c)};
          Emit(j);
          if (ite) {
            Request(Break::Space);
          } else if (!disjunction) {
            doc_.BeginNest(4);
            Request(Break::HardLine);
          }
          (this->*c.unparse)(c.node, true, false);
          if (!ite && !disjunction) {
            doc_.EndNest();
          }
          Request(Break::HardLine);
          Emit(match_[j]);
          if (nested) {
            doc_.EndNest();
          }
          e += 2;
          continue;
        }
      }
      Gap(n, elements, e, block, nests, brackets, backquotes);
    }
    for (; brackets > 0; --brackets) {
      doc_.EndNest();
      doc_.EndGroup();
    }
    for (; nests > 0; --nests) {
      doc_.EndNest();
    }
  }
  doc_.EndGroup();
  if constexpr (isDisjunction<N> || isIfThenElse<N>) {
    if (n.tokens.last + 1 < tokens_.size() &&
        tokens_[n.tokens.last + 1].kind == TokenKind::Close) {
      Request(Break::HardLine);
    }
  }
  if (wrap) {
    EmitSynthetic(")", TokenKind::Close);
  }
}

// A token of the node's own, between its children.
template <typename N>
void CanonicalUnparser::Gap(const N &n, llvm::ArrayRef<Element> elements,
    std::size_t e, bool block, unsigned &nests, unsigned &brackets,
    unsigned &backquotes) {
  std::uint32_t j{elements[e].token};
  const Token &token{tokens_[j]};
  llvm::StringRef spelling{token.spelling};
  switch (token.kind) {
  case TokenKind::Open:
  case TokenKind::OpenCt:
  case TokenKind::OpenList:
  case TokenKind::OpenCurly:
    Emit(j);
    doc_.BeginGroup();
    doc_.BeginNest(4);
    // { Goal } in a DCG body has spaces inside its braces.
    Request(std::is_same_v<N, DcgBraces> ? Break::Line : Break::SoftLine);
    ++brackets;
    return;
  case TokenKind::Close:
  case TokenKind::CloseList:
  case TokenKind::CloseCurly:
    if (brackets > 0) {
      doc_.EndNest();
      if constexpr (std::is_same_v<N, DcgBraces>) {
        Request(Break::Line);
      }
      Emit(j);
      doc_.EndGroup();
      --brackets;
    } else {
      Emit(j);
    }
    return;
  case TokenKind::Comma:
    Emit(j);
    Request(isConjunction<N> && block ? Break::HardLine : Break::Line);
    return;
  case TokenKind::HtSep:
    Request(Break::Line);
    Emit(j);
    return;
  case TokenKind::Backquote:
    Emit(j, backquotes++ % 2 == 0 ? tightAfter : tight);
    return;
  case TokenKind::Name:
    break;
  default:
    Emit(j);
    return;
  }
  if constexpr (isConjunction<N>) {
    if (spelling == "&") {
      Emit(j);
      Request(block ? Break::HardLine : Break::Line);
      return;
    }
  } else if constexpr (isDisjunction<N>) {
    if (spelling == ";") {
      Request(Break::HardLine);
      Emit(j);
      return;
    }
  } else if constexpr (isIfThenElse<N>) {
    if (spelling == (n.form == IfThenElseForm::IfThenElse ? "else" : ";")) {
      Request(Break::HardLine);
      Emit(j);
      return;
    }
  } else if constexpr (std::is_same_v<N, DuTypeDefn>) {
    if (spelling == "--->") {
      doc_.BeginNest(4);
      ++nests;
      Request(Break::HardLine);
      Emit(j);
      Request(Break::Space, 4);
      return;
    }
    if (spelling == ";") {
      Request(Break::HardLine);
      Emit(j);
      Request(Break::Space, 7);
      return;
    }
    if (spelling == "where") {
      Request(Break::HardLine);
      Emit(j);
      return;
    }
  } else if constexpr (std::is_same_v<N, NameAndArity>) {
    if (spelling == "/") {
      Emit(j, tight);
      return;
    }
  }
  if (spelling == "::") {
    Emit(j, tight);
    return;
  }
  // A graphic prefix operator, such as - or \+, keeps the source's layout
  // after it.
  unsigned flags{0};
  if (e == 0 && elements.size() > 1 && elements[1].child &&
      IsGraphicChar(spelling.front())) {
    flags |= keepAdjacencyAfter;
  }
  Emit(j, flags);
}

} // namespace

void UnparseCanonical(llvm::raw_ostream &o, const ParseTree &tree,
    const TokenSequence &tokens, bool parenthesize) {
  CanonicalUnparser unparser{tokens, parenthesize};
  unparser.Unparse(tree);
  unparser.Render(o);
}

} // namespace merkurius::parser
