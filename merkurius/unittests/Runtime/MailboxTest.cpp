//===-- unittests/Runtime/MailboxTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runtime's mailboxes (runtime/mailbox.cpp) under threads, which a
// Mercury program of one process cannot reach yet: a receive that blocks
// and is woken by a send or a close from another thread, timeouts, many
// senders at once, and a selective scan while messages arrive. The
// messages are ints, whose descriptor the test makes, as code generation
// would; test/CodeGen/process-*.m test the copy of other types.
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

using merkurius::runtime::TypeCtor;
using merkurius::runtime::Word;

#define HANDLE_PARAMETERS                                                      \
  std::int64_t nodeHigh, std::int64_t nodeLow, std::int64_t placeAndSlot,      \
      std::int64_t generation, std::int64_t secret

extern "C" {
void merk_node(std::int64_t *high, std::int64_t *low);
void merk_mailbox_new(
    std::int64_t *placeAndSlot, std::int64_t *generation, std::int64_t *secret);
void merk_mailbox_close(HANDLE_PARAMETERS);
void merk_mailbox_send(HANDLE_PARAMETERS, Word typeInfo, Word value);
std::int64_t merk_mailbox_deadline(std::int64_t milliseconds);
std::int64_t merk_mailbox_wait(
    HANDLE_PARAMETERS, std::int64_t position, std::int64_t deadline);
Word merk_mailbox_message(HANDLE_PARAMETERS, std::int64_t position);
void merk_mailbox_remove(HANDLE_PARAMETERS, std::int64_t position);
std::int64_t merk_mailbox_register(
    HANDLE_PARAMETERS, const char *name, Word typeInfo);
void merk_mailbox_unregister(const char *name);
std::int64_t merk_mailbox_lookup(const char *name, Word typeInfo,
    std::int64_t *placeAndSlot, std::int64_t *generation, std::int64_t *secret);
}

namespace {

// int's descriptor: a message of it is its word, which the copy keeps.
const TypeCtor kInt{merkurius::runtime::Int, 0, 0, 0, "builtin", "int",
    nullptr, nullptr, nullptr, {}, nullptr, nullptr};
const TypeCtor kFloat{merkurius::runtime::Float, 0, 0, 0, "builtin", "float",
    nullptr, nullptr, nullptr, {}, nullptr, nullptr};

Word IntType() { return reinterpret_cast<Word>(&kInt); }
Word FloatType() { return reinterpret_cast<Word>(&kFloat); }

// A handle, as process.m's mailbox(M) has it.
struct Box {
  std::int64_t nodeHigh;
  std::int64_t nodeLow;
  std::int64_t slot;
  std::int64_t generation;
  std::int64_t secret;
};

#define BOX(box) box.nodeHigh, box.nodeLow, box.slot, box.generation, box.secret

Box NewBox() {
  Box box;
  merk_node(&box.nodeHigh, &box.nodeLow);
  merk_mailbox_new(&box.slot, &box.generation, &box.secret);
  return box;
}

void Close(Box box) { merk_mailbox_close(BOX(box)); }

void Send(Box box, Word value) {
  merk_mailbox_send(BOX(box), reinterpret_cast<Word>(&kInt), value);
}

// The oldest message, as process.receive takes it with `any`, waiting up to
// `milliseconds`, or forever for -1; none if none arrives.
std::optional<Word> Receive(Box box, std::int64_t milliseconds) {
  if (!merk_mailbox_wait(BOX(box), 0,
          merk_mailbox_deadline(milliseconds))) {
    return std::nullopt;
  }
  Word value{merk_mailbox_message(BOX(box), 0)};
  merk_mailbox_remove(BOX(box), 0);
  return value;
}

using Clock = std::chrono::steady_clock;

long MillisecondsSince(Clock::time_point start) {
  return static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::now() - start)
          .count());
}

void SendLater(Box box, Word value, int milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  Send(box, value);
}

void CloseLater(Box box, int milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  Close(box);
}

TEST(MailboxTest, ReceiveForeverIsWokenBySend) {
  Box box{NewBox()};
  Clock::time_point start{Clock::now()};
  std::thread sender{SendLater, box, Word{7}, 50};
  std::optional<Word> got{Receive(box, -1)};
  sender.join();
  EXPECT_EQ(got, std::optional<Word>{7});
  EXPECT_GE(MillisecondsSince(start), 40);
  Close(box);
}

TEST(MailboxTest, TimedReceiveIsWokenBySend) {
  Box box{NewBox()};
  Clock::time_point start{Clock::now()};
  std::thread sender{SendLater, box, Word{8}, 30};
  std::optional<Word> got{Receive(box, 10000)};
  sender.join();
  EXPECT_EQ(got, std::optional<Word>{8});
  EXPECT_LT(MillisecondsSince(start), 5000);
  Close(box);
}

TEST(MailboxTest, TimeoutWaitsItsTime) {
  Box box{NewBox()};
  Clock::time_point start{Clock::now()};
  EXPECT_EQ(Receive(box, 100), std::nullopt);
  long waited{MillisecondsSince(start)};
  EXPECT_GE(waited, 100);
  EXPECT_LT(waited, 5000);
  Close(box);
}

TEST(MailboxTest, PollDoesNotWait) {
  Box box{NewBox()};
  Clock::time_point start{Clock::now()};
  EXPECT_EQ(Receive(box, 0), std::nullopt);
  EXPECT_LT(MillisecondsSince(start), 1000);
  Close(box);
}

TEST(MailboxTest, CloseWakesReceiveForever) {
  Box box{NewBox()};
  std::thread closer{CloseLater, box, 50};
  EXPECT_EQ(Receive(box, -1), std::nullopt);
  closer.join();
}

TEST(MailboxTest, StaleHandleReachesNoLaterMailbox) {
  Box old{NewBox()};
  Close(old);
  Box fresh{NewBox()};
  // The closed slot is reused, with a new generation.
  EXPECT_EQ(fresh.slot, old.slot);
  EXPECT_NE(fresh.generation, old.generation);
  Send(old, 1);
  EXPECT_EQ(Receive(fresh, 0), std::nullopt);
  EXPECT_EQ(Receive(old, 0), std::nullopt);
  Send(fresh, 2);
  EXPECT_EQ(Receive(fresh, 0), std::optional<Word>{2});
  Close(fresh);
}

// A handle must name this node, its host, and give the mailbox's secret, or
// the message is dropped, as for a closed mailbox.
TEST(MailboxTest, HandleMustMatch) {
  Box box{NewBox()};
  Box wrongSecret{box}, otherNode{box}, otherPlace{box};
  wrongSecret.secret ^= 1;
  otherNode.nodeLow ^= 1;
  otherPlace.slot |= std::int64_t{1} << 48;
  Send(wrongSecret, 1);
  Send(otherNode, 2);
  Send(otherPlace, 3);
  EXPECT_EQ(Receive(box, 0), std::nullopt);
  EXPECT_EQ(Receive(wrongSecret, 0), std::nullopt);
  Send(box, 4);
  EXPECT_EQ(Receive(otherNode, 0), std::nullopt);
  EXPECT_EQ(Receive(box, 0), std::optional<Word>{4});
  Close(box);
}

TEST(MailboxTest, SecretsDiffer) {
  Box a{NewBox()}, b{NewBox()};
  EXPECT_NE(a.secret, b.secret);
  Box c{NewBox()};
  EXPECT_NE(a.secret, c.secret);
  EXPECT_NE(b.secret, c.secret);
  Close(a);
  Close(b);
  Close(c);
}

// A lookup finds a registered mailbox only for its message type.
TEST(MailboxTest, RegisterAndLookup) {
  Box box{NewBox()};
  ASSERT_EQ(merk_mailbox_register(BOX(box), "pong", IntType()), 0);
  Box found{box};
  found.slot = found.generation = found.secret = -1;
  EXPECT_EQ(merk_mailbox_lookup(
                "pong", IntType(), &found.slot, &found.generation, &found.secret),
      0);
  EXPECT_EQ(found.slot, box.slot);
  EXPECT_EQ(found.generation, box.generation);
  EXPECT_EQ(found.secret, box.secret);
  Send(found, 5);
  EXPECT_EQ(Receive(box, 0), std::optional<Word>{5});
  std::int64_t slot, generation, secret;
  EXPECT_EQ(merk_mailbox_lookup("pong", FloatType(), &slot, &generation, &secret),
      2);
  EXPECT_EQ(merk_mailbox_lookup("ping", IntType(), &slot, &generation, &secret),
      1);
  Close(box);
}

// As Erlang's register/2: a name is taken once, a mailbox has one name, and
// closing a mailbox or unregistering its name frees the name.
TEST(MailboxTest, RegisterErrors) {
  Box a{NewBox()}, b{NewBox()};
  ASSERT_EQ(merk_mailbox_register(BOX(a), "server", IntType()), 0);
  EXPECT_EQ(merk_mailbox_register(BOX(b), "server", IntType()), 1);
  EXPECT_EQ(merk_mailbox_register(BOX(a), "other", IntType()), 2);
  Box closed{NewBox()};
  Close(closed);
  EXPECT_EQ(merk_mailbox_register(BOX(closed), "closed", IntType()), 3);
  std::int64_t slot, generation, secret;
  Close(a);
  EXPECT_EQ(
      merk_mailbox_lookup("server", IntType(), &slot, &generation, &secret), 1);
  ASSERT_EQ(merk_mailbox_register(BOX(b), "server", IntType()), 0);
  merk_mailbox_unregister("server");
  EXPECT_EQ(
      merk_mailbox_lookup("server", IntType(), &slot, &generation, &secret), 1);
  EXPECT_EQ(merk_mailbox_register(BOX(b), "again", IntType()), 0);
  Close(b);
}

// The table of mailboxes grows as mailboxes are made; an empty mailbox's
// queue ends at its own head, in the table, so that end must move with it.
TEST(MailboxTest, GrowingTheTableKeepsMailboxes) {
  Box first{NewBox()};
  Box full{NewBox()};
  Send(full, 1);
  std::vector<Box> more;
  for (int i{0}; i < 100; ++i) {
    more.push_back(NewBox());
  }
  Send(first, 2);
  Send(full, 3);
  EXPECT_EQ(Receive(first, 0), std::optional<Word>{2});
  EXPECT_EQ(Receive(full, 0), std::optional<Word>{1});
  EXPECT_EQ(Receive(full, 0), std::optional<Word>{3});
  EXPECT_EQ(Receive(first, 0), std::nullopt);
  for (Box box : more) {
    Send(box, 4);
    EXPECT_EQ(Receive(box, 0), std::optional<Word>{4});
    Close(box);
  }
  Close(first);
  Close(full);
}

// Each sender's messages arrive in the order it sent them, and every
// message arrives once.
void SendNumbered(Box box, Word sender, Word count) {
  for (Word i{0}; i < count; ++i) {
    Send(box, (sender << 32) | i);
  }
}

TEST(MailboxTest, ManySendersKeepTheirOrder) {
  constexpr Word kSenders{8}, kEach{20000};
  Box box{NewBox()};
  std::vector<std::thread> senders;
  for (Word s{0}; s < kSenders; ++s) {
    senders.emplace_back(SendNumbered, box, s, kEach);
  }
  std::vector<Word> next(kSenders, 0);
  bool ordered{true};
  for (Word received{0}; received < kSenders * kEach; ++received) {
    std::optional<Word> got{Receive(box, 10000)};
    ASSERT_TRUE(got.has_value()) << "after " << received << " messages";
    Word sender{*got >> 32}, number{*got & 0xffffffff};
    ASSERT_LT(sender, kSenders);
    ordered &= number == next[sender];
    next[sender] = number + 1;
  }
  for (std::thread &sender : senders) {
    sender.join();
  }
  EXPECT_TRUE(ordered);
  for (Word s{0}; s < kSenders; ++s) {
    EXPECT_EQ(next[s], kEach) << "sender " << s;
  }
  EXPECT_EQ(Receive(box, 0), std::nullopt);
  Close(box);
}

// A selective receive scans by position, waiting for the next message to
// arrive while a sender appends: the first message it accepts is taken
// out, and the ones it passed stay in order.
TEST(MailboxTest, SelectiveScanWhileSending) {
  constexpr Word kCount{5000};
  Box box{NewBox()};
  std::thread sender{SendNumbered, box, Word{0}, kCount};
  std::int64_t position{0};
  std::optional<Word> found;
  while (merk_mailbox_wait(
      BOX(box), position, merk_mailbox_deadline(10000))) {
    Word value{merk_mailbox_message(BOX(box), position)};
    if (value != 0 && value % 1000 == 0) {
      merk_mailbox_remove(BOX(box), position);
      found = value;
      break;
    }
    ++position;
  }
  sender.join();
  EXPECT_EQ(found, std::optional<Word>{1000});
  for (Word expected{0}; expected < kCount; ++expected) {
    if (expected == 1000) {
      continue;
    }
    ASSERT_EQ(Receive(box, 0), std::optional<Word>{expected});
  }
  EXPECT_EQ(Receive(box, 0), std::nullopt);
  Close(box);
}

// Receivers waiting on different mailboxes each get their own mailbox's
// messages, though one condition variable wakes them all.
void ReceiveAll(Box box, Word count, Word *sum) {
  for (Word i{0}; i < count; ++i) {
    std::optional<Word> got{Receive(box, 10000)};
    if (!got) {
      return;
    }
    *sum += *got;
  }
}

TEST(MailboxTest, WaitersOnDifferentMailboxes) {
  constexpr Word kCount{10000};
  Box a{NewBox()}, b{NewBox()};
  Word sumA{0}, sumB{0};
  std::thread receiverA{ReceiveAll, a, kCount, &sumA};
  std::thread receiverB{ReceiveAll, b, kCount, &sumB};
  for (Word i{1}; i <= kCount; ++i) {
    Send(a, i);
    Send(b, 2 * i);
  }
  receiverA.join();
  receiverB.join();
  EXPECT_EQ(sumA, kCount * (kCount + 1) / 2);
  EXPECT_EQ(sumB, kCount * (kCount + 1));
  Close(a);
  Close(b);
}

} // namespace
