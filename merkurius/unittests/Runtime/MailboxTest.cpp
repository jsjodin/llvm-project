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
// senders at once, a selective scan while messages arrive, and a receive
// over several mailboxes, as a selector waits on them. The
// messages are ints, whose descriptor the test makes, as code generation
// would; test/CodeGen/process-*.m test the copy of other types.
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "encode.h"
#include "node.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
void merk_mailbox_new(Word typeInfo, std::int64_t *placeAndSlot,
    std::int64_t *generation, std::int64_t *secret);
void merk_mailbox_close(HANDLE_PARAMETERS);
void merk_mailbox_send(HANDLE_PARAMETERS, Word typeInfo, Word value);
std::int64_t merk_mailbox_deadline(std::int64_t milliseconds);
std::int64_t merk_mailbox_wait(
    HANDLE_PARAMETERS, std::int64_t position, std::int64_t deadline);
std::int64_t merk_mailbox_poll(HANDLE_PARAMETERS, std::int64_t position);
std::int64_t merk_mailbox_arrivals(void);
std::int64_t merk_mailbox_wait_arrival(std::int64_t seen, std::int64_t deadline);
Word merk_mailbox_message(HANDLE_PARAMETERS, std::int64_t position);
void merk_mailbox_remove(HANDLE_PARAMETERS, std::int64_t position);
std::int64_t merk_mailbox_register(HANDLE_PARAMETERS, const char *name);
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

Box NewBox(Word type = IntType()) {
  Box box;
  merk_node(&box.nodeHigh, &box.nodeLow);
  merk_mailbox_new(type, &box.slot, &box.generation, &box.secret);
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

// The oldest message of either mailbox, as process.receive takes it over
// a selector with a `selecting` case for each, waiting up to
// `milliseconds`, or forever for -1; none if none arrives, or both are
// closed. `from` is 0 for a, 1 for b.
std::optional<Word> ReceiveEither(
    Box a, Box b, std::int64_t milliseconds, int *from = nullptr) {
  std::int64_t deadline{merk_mailbox_deadline(milliseconds)};
  for (;;) {
    std::int64_t seen{merk_mailbox_arrivals()};
    std::int64_t stampA{merk_mailbox_poll(BOX(a), 0)};
    std::int64_t stampB{merk_mailbox_poll(BOX(b), 0)};
    if (stampA > 0 && (stampB <= 0 || stampA < stampB)) {
      Word value{merk_mailbox_message(BOX(a), 0)};
      merk_mailbox_remove(BOX(a), 0);
      if (from) {
        *from = 0;
      }
      return value;
    }
    if (stampB > 0) {
      Word value{merk_mailbox_message(BOX(b), 0)};
      merk_mailbox_remove(BOX(b), 0);
      if (from) {
        *from = 1;
      }
      return value;
    }
    if (stampA < 0 && stampB < 0) {
      return std::nullopt;
    }
    if (!merk_mailbox_wait_arrival(seen, deadline)) {
      return std::nullopt;
    }
  }
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
  ASSERT_EQ(merk_mailbox_register(BOX(box), "pong"), 0);
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
  ASSERT_EQ(merk_mailbox_register(BOX(a), "server"), 0);
  EXPECT_EQ(merk_mailbox_register(BOX(b), "server"), 1);
  EXPECT_EQ(merk_mailbox_register(BOX(a), "other"), 2);
  Box closed{NewBox()};
  Close(closed);
  EXPECT_EQ(merk_mailbox_register(BOX(closed), "closed"), 3);
  std::int64_t slot, generation, secret;
  Close(a);
  EXPECT_EQ(
      merk_mailbox_lookup("server", IntType(), &slot, &generation, &secret), 1);
  ASSERT_EQ(merk_mailbox_register(BOX(b), "server"), 0);
  merk_mailbox_unregister("server");
  EXPECT_EQ(
      merk_mailbox_lookup("server", IntType(), &slot, &generation, &secret), 1);
  EXPECT_EQ(merk_mailbox_register(BOX(b), "again"), 0);
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

TEST(MailboxTest, PollGivesStamps) {
  Box a{NewBox()}, b{NewBox()};
  EXPECT_EQ(merk_mailbox_poll(BOX(a), 0), 0);
  Send(a, 1);
  Send(b, 2);
  Send(a, 3);
  std::int64_t first{merk_mailbox_poll(BOX(a), 0)};
  std::int64_t second{merk_mailbox_poll(BOX(b), 0)};
  std::int64_t third{merk_mailbox_poll(BOX(a), 1)};
  EXPECT_GT(first, 0);
  EXPECT_LT(first, second);
  EXPECT_LT(second, third);
  EXPECT_EQ(merk_mailbox_poll(BOX(a), 2), 0);
  Close(a);
  EXPECT_EQ(merk_mailbox_poll(BOX(a), 0), -1);
  EXPECT_EQ(merk_mailbox_poll(BOX(b), 0), second);
  Close(b);
}

TEST(MailboxTest, SelectorIsWokenBySendToEither) {
  Box a{NewBox()}, b{NewBox()};
  Clock::time_point start{Clock::now()};
  std::thread sender{SendLater, b, Word{9}, 50};
  int from{-1};
  std::optional<Word> got{ReceiveEither(a, b, -1, &from)};
  sender.join();
  EXPECT_EQ(got, std::optional<Word>{9});
  EXPECT_EQ(from, 1);
  EXPECT_GE(MillisecondsSince(start), 40);
  Close(a);
  Close(b);
}

// A message to a third mailbox wakes the wait, which goes on until its
// time is up.
TEST(MailboxTest, SelectorTimeoutWaitsItsTime) {
  Box a{NewBox()}, b{NewBox()}, c{NewBox()};
  Clock::time_point start{Clock::now()};
  std::thread sender{SendLater, c, Word{1}, 20};
  EXPECT_EQ(ReceiveEither(a, b, 150), std::nullopt);
  sender.join();
  long waited{MillisecondsSince(start)};
  EXPECT_GE(waited, 150);
  EXPECT_LT(waited, 5000);
  EXPECT_EQ(Receive(c, 0), std::optional<Word>{1});
  Close(a);
  Close(b);
  Close(c);
}

// Closing one mailbox wakes the wait, which goes on for the other; closing
// both ends it.
TEST(MailboxTest, SelectorEndsWhenBothAreClosed) {
  Box a{NewBox()}, b{NewBox()};
  Clock::time_point start{Clock::now()};
  std::thread closerA{CloseLater, a, 20};
  std::thread closerB{CloseLater, b, 80};
  EXPECT_EQ(ReceiveEither(a, b, -1), std::nullopt);
  closerA.join();
  closerB.join();
  EXPECT_GE(MillisecondsSince(start), 70);
}

// One thread sends 1, 2, 3, ... to the two mailboxes in turn while the
// receiver takes the oldest of either, so it must see them in order.
void SendAlternately(Box a, Box b, Word count) {
  for (Word i{1}; i <= count; ++i) {
    Send(i % 2 ? a : b, i);
  }
}

TEST(MailboxTest, SelectorTakesTheOldestWhileSending) {
  constexpr Word kCount{20000};
  Box a{NewBox()}, b{NewBox()};
  std::thread sender{SendAlternately, a, b, kCount};
  Word expected{1};
  Word outOfOrder{0};
  for (; expected <= kCount; ++expected) {
    std::optional<Word> got{ReceiveEither(a, b, 10000)};
    if (!got) {
      break;
    }
    outOfOrder += *got != expected;
  }
  sender.join();
  EXPECT_EQ(expected, kCount + 1);
  EXPECT_EQ(outOfOrder, Word{0});
  Close(a);
  Close(b);
}

merkurius::runtime::Handle HandleOf(Box box) {
  return {static_cast<Word>(box.nodeHigh), static_cast<Word>(box.nodeLow),
      static_cast<Word>(box.slot), box.generation,
      static_cast<Word>(box.secret)};
}

// What a connection's reader does with a message for this node: decoded by
// the mailbox's own type and queued, or dropped as a local send to a closed
// mailbox is, or rejected if it is not a value of that type.
TEST(MailboxTest, DeliverEncoded) {
  using merkurius::runtime::Delivery;
  using merkurius::runtime::DeliverEncoded;
  Box box{NewBox()};
  merkurius::runtime::Encoded encoded;
  ASSERT_EQ(merkurius::runtime::Encode(static_cast<Word>(-42), IntType(),
                encoded),
      nullptr);
  ASSERT_EQ(encoded.size, 8u);
  EXPECT_EQ(DeliverEncoded(HandleOf(box), encoded.bytes, encoded.size),
      Delivery::Delivered);
  EXPECT_EQ(Receive(box, 0), std::optional<Word>{static_cast<Word>(-42)});
  Box wrongSecret{box}, otherPlace{box};
  wrongSecret.secret ^= 1;
  otherPlace.slot |= std::int64_t{1} << 48;
  EXPECT_EQ(DeliverEncoded(HandleOf(wrongSecret), encoded.bytes, encoded.size),
      Delivery::Dropped);
  EXPECT_EQ(DeliverEncoded(HandleOf(otherPlace), encoded.bytes, encoded.size),
      Delivery::Dropped);
  EXPECT_EQ(DeliverEncoded(HandleOf(box), encoded.bytes, 4),
      Delivery::Undecodable);
  std::uint8_t twoItems[16]{};
  EXPECT_EQ(DeliverEncoded(HandleOf(box), twoItems, sizeof twoItems),
      Delivery::Undecodable);
  EXPECT_EQ(Receive(box, 0), std::nullopt);
  Close(box);
  EXPECT_EQ(DeliverEncoded(HandleOf(box), encoded.bytes, encoded.size),
      Delivery::Dropped);
  std::free(encoded.bytes);
}

// Another node's lookup gives the type's fingerprint, which must be the
// registered mailbox's.
TEST(MailboxTest, LookupForPeer) {
  using merkurius::runtime::LookupForPeer;
  Box box{NewBox()};
  ASSERT_EQ(merk_mailbox_register(BOX(box), "peer-pong"), 0);
  std::uint8_t ofInt[merkurius::runtime::kFingerprintBytes];
  std::uint8_t ofFloat[merkurius::runtime::kFingerprintBytes];
  merkurius::runtime::Fingerprint(IntType(), ofInt);
  merkurius::runtime::Fingerprint(FloatType(), ofFloat);
  merkurius::runtime::Handle found{};
  ASSERT_EQ(LookupForPeer("peer-pong", ofInt, found), 0);
  merkurius::runtime::Handle expected{HandleOf(box)};
  EXPECT_EQ(found.nodeHigh, expected.nodeHigh);
  EXPECT_EQ(found.nodeLow, expected.nodeLow);
  EXPECT_EQ(found.placeAndSlot, expected.placeAndSlot);
  EXPECT_EQ(found.generation, expected.generation);
  EXPECT_EQ(found.secret, expected.secret);
  EXPECT_EQ(LookupForPeer("peer-pong", ofFloat, found), 2);
  EXPECT_EQ(LookupForPeer("peer-ping", ofInt, found), 1);
  Close(box);
  EXPECT_EQ(LookupForPeer("peer-pong", ofInt, found), 1);
}

} // namespace
