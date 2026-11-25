//===-- lib/Parser/message.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/lib/Parser/message.cpp
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/message.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace merkurius::parser {

void MessageFormattedText::Format(const MessageFixedText *text, ...) {
  const char *p{text->text().begin()};
  std::string asString;
  if (*text->text().end() != '\0') {
    // not NUL-terminated
    asString = text->text().NULTerminatedToString();
    p = asString.c_str();
  }
  va_list ap;
  va_start(ap, text);
#ifdef _MSC_VER
  // Microsoft has a separate function for "positional arguments", which is
  // used in some messages.
  int need{_vsprintf_p(nullptr, 0, p, ap)};
#else
  int need{vsnprintf(nullptr, 0, p, ap)};
#endif

  CHECK(need >= 0);
  char *buffer{
      static_cast<char *>(std::malloc(static_cast<std::size_t>(need) + 1))};
  CHECK(buffer);
  va_end(ap);
  va_start(ap, text);
#ifdef _MSC_VER
  // Use positional argument variant of printf.
  int need2{_vsprintf_p(buffer, need + 1, p, ap)};
#else
  int need2{vsnprintf(buffer, need + 1, p, ap)};
#endif
  CHECK(need2 == need);
  va_end(ap);
  string_ = buffer;
  std::free(buffer);
  conversions_.clear();
}

const char *MessageFormattedText::Convert(const std::string &s) {
  conversions_.emplace_front(s);
  return conversions_.front().c_str();
}

const char *MessageFormattedText::Convert(std::string &&s) {
  conversions_.emplace_front(std::move(s));
  return conversions_.front().c_str();
}

const char *MessageFormattedText::Convert(llvm::StringRef x) {
  return Convert(x.str());
}

bool Message::SortBefore(const Message &that) const {
  // Messages may be located by ProvenanceRange or by CharBlock; those with
  // ProvenanceRange locations are ordered before others for sorting.
  return common::visit(
      common::visitors{
          [](CharBlock cb1, CharBlock cb2) {
            return cb1.begin() < cb2.begin();
          },
          [](CharBlock, const ProvenanceRange &) { return false; },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() < pr2.start();
          },
          [](const ProvenanceRange &, CharBlock) { return true; },
      },
      location_, that.location_);
}

bool Message::IsFatal() const { return IsFatalSeverity(severity()); }

Severity Message::severity() const {
  return common::visit(
      common::visitors{
          [](const MessageFixedText &x) { return x.severity(); },
          [](const MessageFormattedText &x) { return x.severity(); },
      },
      text_);
}

std::string Message::ToString() const {
  return common::visit(
      common::visitors{
          [](const MessageFixedText &t) {
            return t.text().NULTerminatedToString();
          },
          [](const MessageFormattedText &t) { return t.string(); },
      },
      text_);
}

std::optional<ProvenanceRange> Message::GetProvenanceRange(
    const AllSources &allSources) const {
  return common::visit(
      common::visitors{
          [&](CharBlock cb) { return allSources.GetProvenanceRange(cb); },
          [](const ProvenanceRange &pr) { return std::make_optional(pr); },
      },
      location_);
}

static std::string Prefix(Severity severity) {
  switch (severity) {
  case Severity::Error:
    return "error: ";
  case Severity::Warning:
    return "warning: ";
  case Severity::Because:
    return "because: ";
  case Severity::Todo:
    return "error: not yet implemented: ";
  case Severity::None:
    break;
  }
  return "";
}

void Message::Emit(llvm::raw_ostream &o, const AllSources &sources,
    bool echoSourceLine) const {
  std::optional<ProvenanceRange> provenanceRange{GetProvenanceRange(sources)};
  sources.EmitMessage(
      o, provenanceRange, ToString(), Prefix(severity()), echoSourceLine);
  for (const Message *attachment{attachment_.get()}; attachment;
      attachment = attachment->attachment_.get()) {
    sources.EmitMessage(o, attachment->GetProvenanceRange(sources),
        attachment->ToString(), Prefix(attachment->severity()), echoSourceLine);
  }
}

// Messages are equal if they're for the same location and text, and the user
// visible aspects of their attachments are the same
bool Message::operator==(const Message &that) const {
  if (!AtSameLocation(that) || ToString() != that.ToString() ||
      severity() != that.severity()) {
    return false;
  }
  const Message *thatAttachment{that.attachment_.get()};
  for (const Message *attachment{attachment_.get()}; attachment;
       attachment = attachment->attachment_.get()) {
    if (!thatAttachment || !attachment->AtSameLocation(*thatAttachment) ||
        attachment->ToString() != thatAttachment->ToString() ||
        attachment->severity() != thatAttachment->severity()) {
      return false;
    }
    thatAttachment = thatAttachment->attachment_.get();
  }
  return !thatAttachment;
}

Message &Message::Attach(Message *m) {
  if (!attachment_) {
    attachment_ = m;
  } else {
    if (attachment_->references() > 1) {
      // Don't attach to a shared context attachment; copy it first.
      attachment_ = new Message{*attachment_};
    }
    attachment_->Attach(m);
  }
  return *this;
}

bool Message::AtSameLocation(const Message &that) const {
  return common::visit(
      common::visitors{
          [](CharBlock cb1, CharBlock cb2) {
            return cb1.begin() == cb2.begin();
          },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() == pr2.start();
          },
          [](const auto &, const auto &) { return false; },
      },
      location_, that.location_);
}

void Messages::Emit(llvm::raw_ostream &o, const AllSources &allSources,
    bool echoSourceLines, std::size_t maxErrorsToEmit,
    bool warningsAreErrors) const {
  std::vector<const Message *> sorted;
  for (const auto &msg : messages_) {
    sorted.push_back(&msg);
  }
  std::stable_sort(sorted.begin(), sorted.end(),
      [](const Message *x, const Message *y) { return x->SortBefore(*y); });
  std::vector<const Message *> msgsWithLastLocation;
  std::size_t errorsEmitted{0};
  for (const Message *msg : sorted) {
    bool shouldSkipMsg{false};
    // Don't emit two identical messages for the same location.
    // At the same location, messages are sorted by the order they were
    // added to the Messages buffer, which is a decent proxy for the
    // causality of the messages.
    if (!msgsWithLastLocation.empty()) {
      if (msgsWithLastLocation[0]->AtSameLocation(*msg)) {
        for (const Message *msgAtThisLocation : msgsWithLastLocation) {
          if (*msg == *msgAtThisLocation) {
            shouldSkipMsg = true; // continue loop over sorted messages
            break;
          }
        }
      } else {
        msgsWithLastLocation.clear();
      }
    }
    if (shouldSkipMsg) {
      continue;
    }
    msgsWithLastLocation.push_back(msg);
    msg->Emit(o, allSources, echoSourceLines);
    if (warningsAreErrors || msg->IsFatal()) {
      ++errorsEmitted;
    }
    // If maxErrorsToEmit is 0, emit all errors, otherwise break after
    // maxErrorsToEmit.
    if (maxErrorsToEmit > 0 && errorsEmitted >= maxErrorsToEmit) {
      break;
    }
  }
}

bool Messages::AnyFatalError(bool warningsAreErrors) const {
  // Short-circuit in the most common case.
  if (messages_.empty()) {
    return false;
  }
  // If warnings are errors and there are warnings or errors, this is fatal.
  // This preserves the compiler's current behavior of treating any non-fatal
  // message as a warning. We may want to refine this in the future.
  if (warningsAreErrors) {
    return true;
  }
  // Otherwise, check the message buffer for fatal errors.
  for (const auto &msg : messages_) {
    if (msg.IsFatal()) {
      return true;
    }
  }
  return false;
}
} // namespace merkurius::parser
