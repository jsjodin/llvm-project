//===-- include/merkurius/Parser/message.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/include/flang/Parser/message.h
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_MESSAGE_H_
#define MERKURIUS_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.
//
// Changes from Flang: Fortran's language-feature and usage-warning controls,
// the ErrorUnlessDeadCode and Portability severities, and the merged
// "expected" texts of Flang's character-level parser are removed. Messages
// are located through AllSources, since there is no cooked source.

#include "char-block.h"
#include "provenance.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Common/reference-counted.h"
#include "merkurius/Common/restorer.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <forward_list>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace merkurius::parser {

// Use "..."_err_en_US, "..."_warn_en_US, "..."_because_en_US,
// "..."_todo_en_US, and "..."_en_US string literals to define the static text
// and severity of a message or attachment.
enum class Severity {
  Error, // fatal error that prevents code and interface file generation
  Warning, // likely problem
  Because, // for AttachTo(), explanatory attachment to support another message
  Context, // (internal): attachment from SetContext()
  Todo, // a feature that's not yet implemented, a fatal error
  None // everything else, common for attachments with source locations
};

inline constexpr bool IsFatalSeverity(Severity severity) {
  return severity == Severity::Error || severity == Severity::Todo;
}

inline constexpr bool IsWarningSeverity(Severity severity) {
  return severity == Severity::Warning;
}

class MessageFixedText {
public:
  constexpr MessageFixedText() {}
  constexpr MessageFixedText(
      const char str[], std::size_t n, Severity severity = Severity::None)
      : text_{str, n}, severity_{severity} {}
  constexpr MessageFixedText(const MessageFixedText &) = default;
  constexpr MessageFixedText(MessageFixedText &&) = default;
  constexpr MessageFixedText &operator=(const MessageFixedText &) = default;
  constexpr MessageFixedText &operator=(MessageFixedText &&) = default;

  CharBlock text() const { return text_; }
  bool empty() const { return text_.empty(); }
  Severity severity() const { return severity_; }
  MessageFixedText &set_severity(Severity severity) {
    severity_ = severity;
    return *this;
  }
  bool IsFatal() const { return IsFatalSeverity(severity_); }

private:
  CharBlock text_;
  Severity severity_{Severity::None};
};

inline namespace literals {
constexpr MessageFixedText operator""_err_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Error};
}
constexpr MessageFixedText operator""_warn_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Warning};
}
constexpr MessageFixedText operator""_because_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Because};
}
constexpr MessageFixedText operator""_todo_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Todo};
}
constexpr MessageFixedText operator""_en_US(const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::None};
}
} // namespace literals

// The construction of a MessageFormattedText uses a MessageFixedText
// as a vsnprintf() formatting string that is applied to the
// following arguments.  CharBlock, std::string, std::string_view, and
// llvm::StringRef argument values are also supported; they are automatically
// converted into char pointers that are suitable for '%s' formatting.
class MessageFormattedText {
public:
  template <typename... A>
  MessageFormattedText(const MessageFixedText &text, A &&...x)
      : severity_{text.severity()} {
    Format(&text, Convert(std::forward<A>(x))...);
  }
  MessageFormattedText(const MessageFormattedText &) = default;
  MessageFormattedText(MessageFormattedText &&) = default;
  MessageFormattedText &operator=(const MessageFormattedText &) = default;
  MessageFormattedText &operator=(MessageFormattedText &&) = default;
  const std::string &string() const { return string_; }
  bool IsFatal() const { return IsFatalSeverity(severity_); }
  Severity severity() const { return severity_; }
  MessageFormattedText &set_severity(Severity severity) {
    severity_ = severity;
    return *this;
  }
  std::string MoveString() { return std::move(string_); }
  bool operator==(const MessageFormattedText &that) const {
    return severity_ == that.severity_ && string_ == that.string_;
  }
  bool operator!=(const MessageFormattedText &that) const {
    return !(*this == that);
  }

private:
  void Format(const MessageFixedText *, ...);

  template <typename A> A Convert(const A &x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return x;
  }
  template <typename A> common::IfNoLvalue<A, A> Convert(A &&x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return std::move(x);
  }
  const char *Convert(const char *s) { return s; }
  const char *Convert(char *s) { return s; }
  const char *Convert(const std::string &);
  const char *Convert(std::string &&);
  const char *Convert(const std::string_view &);
  const char *Convert(std::string_view &&);
  const char *Convert(CharBlock);
  const char *Convert(llvm::StringRef);
  std::intmax_t Convert(std::int64_t x) { return x; }
  std::uintmax_t Convert(std::uint64_t x) { return x; }

  Severity severity_;
  std::string string_;
  std::forward_list<std::string> conversions_; // preserves created strings
};

class Message : public common::ReferenceCounted<Message> {
public:
  using Reference = common::CountedReference<Message>;

  Message(const Message &) = default;
  Message(Message &&) = default;
  Message &operator=(const Message &) = default;
  Message &operator=(Message &&) = default;

  Message(ProvenanceRange pr, const MessageFixedText &t)
      : location_{pr}, text_{t} {}
  Message(ProvenanceRange pr, const MessageFormattedText &s)
      : location_{pr}, text_{s} {}
  Message(ProvenanceRange pr, MessageFormattedText &&s)
      : location_{pr}, text_{std::move(s)} {}

  Message(CharBlock csr, const MessageFixedText &t)
      : location_{csr}, text_{t} {}
  Message(CharBlock csr, const MessageFormattedText &s)
      : location_{csr}, text_{s} {}
  Message(CharBlock csr, MessageFormattedText &&s)
      : location_{csr}, text_{std::move(s)} {}

  template <typename RANGE, typename A, typename... As>
  Message(RANGE r, const MessageFixedText &t, A &&x, As &&...xs)
      : location_{r}, text_{MessageFormattedText{
                          t, std::forward<A>(x), std::forward<As>(xs)...}} {}

  Reference attachment() const { return attachment_; }

  void SetContext(Message *c) {
    attachment_ = c;
    attachmentIsContext_ = true;
  }
  Message &Attach(Message *);
  Message &Attach(std::unique_ptr<Message> &&);
  template <typename... A> Message &Attach(A &&...args) {
    return Attach(new Message{std::forward<A>(args)...}); // reference-counted
  }

  bool SortBefore(const Message &that) const;
  bool IsFatal() const;
  Severity severity() const;
  Message &set_severity(Severity);
  std::string ToString() const;
  std::optional<ProvenanceRange> GetProvenanceRange(const AllSources &) const;
  void Emit(llvm::raw_ostream &, const AllSources &,
      bool echoSourceLine = true) const;

  // If this Message or any of its attachments locates itself via a CharBlock,
  // replace its location with the corresponding ProvenanceRange.
  void ResolveProvenances(const AllSources &);

  bool operator==(const Message &that) const;
  bool operator!=(const Message &that) const { return !(*this == that); }
  bool AtSameLocation(const Message &) const;

private:
  std::variant<ProvenanceRange, CharBlock> location_;
  std::variant<MessageFixedText, MessageFormattedText> text_;
  bool attachmentIsContext_{false};
  Reference attachment_;
};

class Messages {
public:
  Messages() {}
  Messages(Messages &&that) : messages_{std::move(that.messages_)} {}
  Messages &operator=(Messages &&that) {
    messages_ = std::move(that.messages_);
    return *this;
  }

  std::list<Message> &messages() { return messages_; }
  const std::list<Message> &messages() const { return messages_; }
  bool empty() const { return messages_.empty(); }
  void clear() { messages_.clear(); }

  template <typename... A> Message &Say(A &&...args) {
    return messages_.emplace_back(std::forward<A>(args)...);
  }

  void Annex(Messages &&that) {
    messages_.splice(messages_.end(), that.messages_);
  }

  void Copy(const Messages &);
  void ResolveProvenances(const AllSources &);
  void Emit(llvm::raw_ostream &, const AllSources &,
      bool echoSourceLines = true, std::size_t maxErrorsToEmit = 0,
      bool warningsAreErrors = false) const;
  void AttachTo(Message &, std::optional<Severity> = std::nullopt);
  bool AnyFatalError(bool warningsAreErrors = false) const;

private:
  std::list<Message> messages_;
};

class ContextualMessages {
public:
  ContextualMessages() = default;
  ContextualMessages(CharBlock at, Messages *m) : at_{at}, messages_{m} {}
  explicit ContextualMessages(Messages *m) : messages_{m} {}
  ContextualMessages(const ContextualMessages &that)
      : at_{that.at_}, messages_{that.messages_} {}

  CharBlock at() const { return at_; }
  Messages *messages() const { return messages_; }
  Message::Reference contextMessage() const { return contextMessage_; }
  bool empty() const { return !messages_ || messages_->empty(); }

  // Set CharBlock for messages; restore when the returned value is deleted
  common::Restorer<CharBlock> SetLocation(CharBlock at) {
    if (at.empty()) {
      at = at_;
    }
    return common::ScopedSet(at_, std::move(at));
  }

  common::Restorer<Message::Reference> SetContext(Message *m) {
    if (!m) {
      m = contextMessage_.get();
    }
    return common::ScopedSet(contextMessage_, m);
  }

  // Diverts messages to another buffer; restored when the returned
  // value is deleted.
  common::Restorer<Messages *> SetMessages(Messages &buffer) {
    return common::ScopedSet(messages_, &buffer);
  }
  // Discard future messages until the returned value is deleted.
  common::Restorer<Messages *> DiscardMessages() {
    return common::ScopedSet(messages_, nullptr);
  }

  template <typename... A> Message *Say(A &&...args) {
    return Say(at_, std::forward<A>(args)...);
  }

  template <typename... A> Message *Say(CharBlock at, A &&...args) {
    if (messages_ != nullptr) {
      auto &msg{messages_->Say(at, std::forward<A>(args)...)};
      if (contextMessage_) {
        msg.SetContext(contextMessage_.get());
      }
      return &msg;
    } else {
      return nullptr;
    }
  }

  template <typename... A>
  Message *Say(std::optional<CharBlock> at, A &&...args) {
    return Say(at.value_or(at_), std::forward<A>(args)...);
  }

  Message *Say(Message &&msg) {
    if (messages_ != nullptr) {
      if (contextMessage_) {
        msg.SetContext(contextMessage_.get());
      }
      return &messages_->Say(std::move(msg));
    } else {
      return nullptr;
    }
  }

private:
  CharBlock at_;
  Messages *messages_{nullptr};
  Message::Reference contextMessage_;
};
} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_MESSAGE_H_
