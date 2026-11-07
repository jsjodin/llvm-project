//===-- runtime/mailbox.cpp - Mailboxes -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mailboxes, under library/process.m (design document, Runtime system,
// "Message passing"). A mailbox is a slot of a table, named by its number and
// the generation the slot had when the mailbox was made, so that the handle of
// a closed mailbox stays dead when the slot is reused. Its handle also names
// the node, by an id chosen at random when the runtime starts, and the place
// within it, 0 for the host, and carries a secret chosen at random when the
// mailbox is made, which a message from another node must give (design
// document, Handles, routing and places). A mailbox may be registered under
// a name, which a lookup finds, if its message type is the one asked for.
// A send copies the
// message by its type_info into a fragment of its own, outside the heap, and
// queues it; the receiver reads it there, so the fragment joins its heap.
//
// A send to a handle of another node encodes the message and routes it
// (node.cpp); a message from another node is decoded by the mailbox's own
// type, which each mailbox keeps from when it was made.
//
// Each message has a stamp in the order of arrival over all mailboxes, by
// which a receive over a selector takes the oldest message that some case
// accepts, and a count of arrivals and closes lets it wait for any of its
// mailboxes with nothing to register.
//
// Temporary (chunk 2, docs/INPROGRESS.md): one lock and one condition
// variable for every mailbox, and fragments never freed, while a program is
// one process; chunk 2 gives each process its mailboxes, a lock-free queue
// and a wake-up of its own, and chunk 3's collector merges fragments into the
// receiver's heap.
//
//===----------------------------------------------------------------------===//

#include "encode.h"
#include "fragment.h"
#include "node.h"
#include "walk.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/random.h>

namespace {

using namespace merkurius::runtime;

[[noreturn]] void Fail(const char *message, const char *detail) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%s\n", message, detail);
  std::exit(1);
}

Word CopyTyped(Word value, const Type &type, Fragment &fragment);

Word CopyValue(Word value, Word type, const Environment *environment,
    Fragment &fragment) {
  Type resolved{Resolve(type, environment)};
  return CopyTyped(value, resolved, fragment);
}

// A term of constructors: its cell copied, field by field, and its tag kept.
// A field of the term's own type is copied in the loop rather than by a
// call, so that the length of a list uses no stack.
Word CopyTerm(Word value, const Type &type, Fragment &fragment) {
  Word result{0};
  Word *slot{&result};
  for (;;) {
    const Word *fields{nullptr};
    Word term{value};
    const Functor *functor{FunctorOf(term, type.ctor, fields)};
    if (!fields || fields == &term) {
      // A constant, or a notag's or a tuple of one's only field, which is
      // the value itself.
      if (!fields) {
        *slot = value;
        return result;
      }
      *slot = CopyValue(value, functor->fields[0], &type.environment, fragment);
      return result;
    }
    if (functor->hidden != 0) {
      Fail("cannot send a value with an existentially typed constructor yet: ",
          functor->name);
    }
    unsigned tag{type.ctor->kind == Tuple
            ? functor->tag
            : static_cast<unsigned>(value & kTagMask)};
    // A constructor that shares its tag has its secondary tag in word 0.
    const Word *from{reinterpret_cast<const Word *>(value - tag)};
    unsigned before{static_cast<unsigned>(fields - from)};
    unsigned arity{functor->arity};
    Word *cell{static_cast<Word *>(
        fragment.Allocate(sizeof(Word) * (before + arity)))};
    for (unsigned i{0}; i < before; ++i) {
      cell[i] = from[i];
    }
    *slot = reinterpret_cast<Word>(cell) + tag;
    Word *copied{cell + before};
    if (arity == 0) {
      return result;
    }
    for (unsigned i{0}; i + 1 < arity; ++i) {
      copied[i] =
          CopyValue(fields[i], functor->fields[i], &type.environment, fragment);
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      copied[arity - 1] =
          CopyValue(fields[arity - 1], last, &type.environment, fragment);
      return result;
    }
    value = fields[arity - 1];
    slot = &copied[arity - 1];
  }
}

Word CopyTyped(Word value, const Type &type, Fragment &fragment) {
  switch (type.ctor->kind) {
  case Int:
  case Uint:
  case Int8:
  case Int16:
  case Int32:
  case Int64:
  case Uint8:
  case Uint16:
  case Uint32:
  case Uint64:
  case Float:
  case Char:
  case Enum:
  case Dummy:
    return value;
  case String: {
    const char *text{reinterpret_cast<const char *>(value)};
    std::size_t bytes{std::strlen(text) + 1};
    char *copy{static_cast<char *>(fragment.Allocate(bytes))};
    std::memcpy(copy, text, bytes);
    return reinterpret_cast<Word>(copy);
  }
  case Array: {
    const Word *array{reinterpret_cast<const Word *>(value)};
    Word size{array[0]};
    Word *copy{
        static_cast<Word *>(fragment.Allocate(sizeof(Word) * (1 + size)))};
    copy[0] = size;
    Word element{(Word{0} << kTagBits) | kVariableTag};
    for (Word i{0}; i < size; ++i) {
      copy[1 + i] =
          CopyValue(array[1 + i], element, &type.environment, fragment);
    }
    return reinterpret_cast<Word>(copy);
  }
  case NoTag:
  case Tuple:
  case Cells:
    return CopyTerm(value, type, fragment);
  case Closure:
    Fail("cannot send a closure yet", "");
  case Void:
    // Nothing has a value of this type.
    std::abort();
  }
  Fail("cannot send a value of type ", type.ctor->name);
}

struct Message {
  Message *next;
  Word value;
  // Its place in the order of arrival over all mailboxes.
  std::uint64_t stamp;
};

// A word from the kernel's source of random numbers.
Word Random() {
  Word word;
  while (getrandom(&word, sizeof word, 0) != sizeof word) {
  }
  return word;
}

// A type_info kept beyond the call that gave it, in memory of its own, as a
// registered mailbox's message type is: its cells copied, its descriptors
// static.
Word KeepTypeInfo(Word info) {
  if ((info & kTagMask) == 0) {
    return info;
  }
  const Word *cell{reinterpret_cast<const Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word arity{counted ? cell[1] : ctor->arity};
  Word before{counted ? Word{2} : Word{1}};
  auto *copy{static_cast<Word *>(Malloc(sizeof(Word) * (before + arity)))};
  for (Word i{0}; i < before; ++i) {
    copy[i] = cell[i];
  }
  for (Word i{0}; i < arity; ++i) {
    copy[before + i] = KeepTypeInfo(cell[before + i]);
  }
  return reinterpret_cast<Word>(copy) + 1;
}

void FreeTypeInfo(Word info) {
  if ((info & kTagMask) == 0) {
    return;
  }
  Word *cell{reinterpret_cast<Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word arity{counted ? cell[1] : ctor->arity};
  Word before{counted ? Word{2} : Word{1}};
  for (Word i{0}; i < arity; ++i) {
    FreeTypeInfo(cell[before + i]);
  }
  std::free(cell);
}

struct Mailbox {
  std::int64_t generation;
  Word secret;
  bool open;
  // Its message type, kept, which decodes messages from other nodes and
  // which a lookup checks.
  Word messageType;
  // The name it is registered under, or null.
  char *name;
  Message *head;
  Message **tail;
  std::int64_t count;
  // The next closed slot, to reuse, or -1.
  std::int64_t nextFree;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t nodeOnce = PTHREAD_ONCE_INIT;
Word nodeHigh{0};
Word nodeLow{0};

void ChooseNode() {
  nodeHigh = Random();
  nodeLow = Random();
}
pthread_cond_t arrived;
pthread_once_t once = PTHREAD_ONCE_INIT;
// The last message's stamp, and the number of arrivals and closes, under
// the lock.
std::uint64_t stamps{0};
std::int64_t arrivals{0};
Mailbox *slots{nullptr};
std::int64_t slotCount{0};
std::int64_t slotCapacity{0};
std::int64_t firstFree{-1};

// Waits on the monotonic clock, which a change of the time of day does not
// move.
void InitializeCondition() {
  pthread_condattr_t attributes;
  pthread_condattr_init(&attributes);
  pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC);
  pthread_cond_init(&arrived, &attributes);
  pthread_condattr_destroy(&attributes);
}

std::int64_t Now() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::int64_t>(now.tv_sec) * 1000000000 + now.tv_nsec;
}

// The mailbox of a handle, or null for a closed one, or one of another node
// or place. Called with the lock held.
Mailbox *Find(const Handle &handle) {
  pthread_once(&nodeOnce, ChooseNode);
  if (handle.nodeHigh != nodeHigh || handle.nodeLow != nodeLow ||
      handle.placeAndSlot >> kPlaceShift != 0) {
    return nullptr;
  }
  Word slot{handle.placeAndSlot & kSlotMask};
  if (slot >= static_cast<Word>(slotCount)) {
    return nullptr;
  }
  Mailbox *mailbox{&slots[slot]};
  return mailbox->open && mailbox->generation == handle.generation &&
          mailbox->secret == handle.secret
      ? mailbox
      : nullptr;
}

// Its name goes, as when it is closed.
void Unregister(Mailbox *mailbox) {
  std::free(mailbox->name);
  mailbox->name = nullptr;
}

// Queues a message whose cells are in a fragment of their own. Called with
// the lock held.
void Queue(Mailbox *mailbox, Message *message) {
  message->stamp = ++stamps;
  *mailbox->tail = message;
  mailbox->tail = &message->next;
  ++mailbox->count;
  ++arrivals;
  pthread_cond_broadcast(&arrived);
}

// Waits on `arrived` until the deadline, or for ever if it is negative:
// false once the deadline has passed. Called with the lock held.
bool WaitArrival(std::int64_t deadline) {
  if (deadline < 0) {
    pthread_cond_wait(&arrived, &lock);
    return true;
  }
  if (Now() >= deadline) {
    return false;
  }
  timespec until{static_cast<time_t>(deadline / 1000000000),
      static_cast<long>(deadline % 1000000000)};
  pthread_cond_timedwait(&arrived, &lock, &until);
  return true;
}

Message *At(Mailbox *mailbox, std::int64_t position) {
  Message *message{mailbox->head};
  for (; position > 0; --position) {
    message = message->next;
  }
  return message;
}

} // namespace

namespace merkurius::runtime {

void ThisNode(Word &high, Word &low) {
  pthread_once(&nodeOnce, ChooseNode);
  high = nodeHigh;
  low = nodeLow;
}

bool IsThisNode(Word high, Word low) {
  pthread_once(&nodeOnce, ChooseNode);
  return high == nodeHigh && low == nodeLow;
}

Delivery DeliverEncoded(
    const Handle &handle, const std::uint8_t *bytes, std::size_t size) {
  pthread_once(&once, InitializeCondition);
  // Decoded with the lock held, so that the mailbox, and its type, stay
  // while it is.
  pthread_mutex_lock(&lock);
  Delivery delivery{Delivery::Dropped};
  if (Mailbox *mailbox{Find(handle)}) {
    Fragment fragment;
    auto *message{static_cast<Message *>(fragment.Allocate(sizeof(Message)))};
    message->next = nullptr;
    if (Decode(bytes, size, mailbox->messageType, fragment, message->value)) {
      delivery = Delivery::Undecodable;
    } else {
      Queue(mailbox, message);
      delivery = Delivery::Delivered;
    }
  }
  pthread_mutex_unlock(&lock);
  return delivery;
}

std::int64_t LookupForPeer(const char *name,
    const std::uint8_t fingerprint[kFingerprintBytes], Handle &found) {
  pthread_mutex_lock(&lock);
  std::int64_t result{1};
  for (std::int64_t i{0}; i < slotCount; ++i) {
    Mailbox &mailbox{slots[i]};
    if (mailbox.open && mailbox.name && std::strcmp(mailbox.name, name) == 0) {
      std::uint8_t own[kFingerprintBytes];
      Fingerprint(mailbox.messageType, own);
      result = std::memcmp(own, fingerprint, kFingerprintBytes) == 0 ? 0 : 2;
      if (result == 0) {
        found = {nodeHigh, nodeLow, static_cast<Word>(i), mailbox.generation,
            mailbox.secret};
      }
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

} // namespace merkurius::runtime

// This node's id, chosen at random the first time it is asked for.
extern "C" void merk_node(std::int64_t *high, std::int64_t *low) {
  Word h, l;
  ThisNode(h, l);
  *high = static_cast<std::int64_t>(h);
  *low = static_cast<std::int64_t>(l);
}

// A new mailbox of this node's host for messages of the type: its slot, with
// place 0, its generation and its secret; merk_node gives the rest of its
// handle.
extern "C" void merk_mailbox_new(Word typeInfo, std::int64_t *placeAndSlot,
    std::int64_t *generation, std::int64_t *secret) {
  pthread_once(&nodeOnce, ChooseNode);
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t index{firstFree};
  if (index >= 0) {
    firstFree = slots[index].nextFree;
  } else {
    if (slotCount == slotCapacity) {
      slotCapacity = slotCapacity ? 2 * slotCapacity : 16;
      slots = static_cast<Mailbox *>(
          std::realloc(slots, sizeof(Mailbox) * slotCapacity));
      if (!slots) {
        Fail("out of memory", "");
      }
      // An empty mailbox's queue ends at its own head, which has moved.
      for (std::int64_t i{0}; i < slotCount; ++i) {
        if (!slots[i].head) {
          slots[i].tail = &slots[i].head;
        }
      }
    }
    index = slotCount++;
    slots[index].generation = 0;
  }
  Mailbox &mailbox{slots[index]};
  mailbox.open = true;
  mailbox.secret = Random();
  mailbox.messageType = KeepTypeInfo(typeInfo);
  mailbox.name = nullptr;
  mailbox.head = nullptr;
  mailbox.tail = &mailbox.head;
  mailbox.count = 0;
  mailbox.nextFree = -1;
  *placeAndSlot = index;
  *generation = mailbox.generation;
  *secret = static_cast<std::int64_t>(mailbox.secret);
  pthread_mutex_unlock(&lock);
}

#define MERK_HANDLE_PARAMETERS                                                 \
  std::int64_t nodeHigh, std::int64_t nodeLow, std::int64_t placeAndSlot,      \
      std::int64_t generation, std::int64_t secret
#define MERK_HANDLE                                                            \
  Handle {                                                                     \
    static_cast<Word>(nodeHigh), static_cast<Word>(nodeLow),                   \
        static_cast<Word>(placeAndSlot), generation, static_cast<Word>(secret) \
  }

// Its queued messages are dropped, its name goes, and its slot gets a new
// generation, so that its handle stays dead when the slot is reused. A
// receive waiting on it wakes and finds nothing.
extern "C" void merk_mailbox_close(MERK_HANDLE_PARAMETERS) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    Unregister(mailbox);
    FreeTypeInfo(mailbox->messageType);
    mailbox->messageType = 0;
    mailbox->open = false;
    ++mailbox->generation;
    mailbox->head = nullptr;
    mailbox->tail = &mailbox->head;
    mailbox->count = 0;
    mailbox->nextFree = firstFree;
    firstFree = mailbox - slots;
    ++arrivals;
    pthread_cond_broadcast(&arrived);
  }
  pthread_mutex_unlock(&lock);
}

// The copy is made before the lock is taken, since it reads only the
// sender's cells; a message to a closed mailbox is then dropped. A message
// to another node's mailbox is encoded, and routed to its connection.
extern "C" void merk_mailbox_send(
    MERK_HANDLE_PARAMETERS, Word typeInfo, Word value) {
  pthread_once(&once, InitializeCondition);
  Handle handle{MERK_HANDLE};
  if (!IsThisNode(handle.nodeHigh, handle.nodeLow)) {
    Encoded encoded;
    if (const char *error{Encode(value, typeInfo, encoded)}) {
      Fail(error, "");
    }
    Route(handle, encoded.bytes, encoded.size);
    std::free(encoded.bytes);
    return;
  }
  Fragment fragment;
  auto *message{static_cast<Message *>(fragment.Allocate(sizeof(Message)))};
  message->next = nullptr;
  message->value = CopyValue(value, typeInfo, nullptr, fragment);
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(handle)}) {
    Queue(mailbox, message);
  }
  pthread_mutex_unlock(&lock);
}

// When a receive of this many milliseconds stops waiting, on the monotonic
// clock in nanoseconds, or -1 for a negative number, which waits forever.
extern "C" std::int64_t merk_mailbox_deadline(std::int64_t milliseconds) {
  return milliseconds < 0 ? -1 : Now() + milliseconds * 1000000;
}

// 1 once the mailbox has a message at `position`, 0 if the deadline passes
// first, or the mailbox is closed or not this node's host's.
extern "C" std::int64_t merk_mailbox_wait(
    MERK_HANDLE_PARAMETERS, std::int64_t position, std::int64_t deadline) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t result{0};
  for (;;) {
    Mailbox *mailbox{Find(MERK_HANDLE)};
    if (!mailbox) {
      break;
    }
    if (mailbox->count > position) {
      result = 1;
      break;
    }
    if (!WaitArrival(deadline)) {
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

// For a receive over a selector, without waiting: the stamp of the
// message at `position`, which is positive, 0 if the mailbox has no message
// there yet, or -1 if it is closed or not this node's host's.
extern "C" std::int64_t merk_mailbox_poll(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  Mailbox *mailbox{Find(MERK_HANDLE)};
  std::int64_t result{!mailbox                    ? -1
          : mailbox->count > position
          ? static_cast<std::int64_t>(At(mailbox, position)->stamp)
          : 0};
  pthread_mutex_unlock(&lock);
  return result;
}

// The number of messages queued and mailboxes closed so far, which a
// receive over a selector reads before it looks at its mailboxes.
extern "C" std::int64_t merk_mailbox_arrivals(void) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t count{arrivals};
  pthread_mutex_unlock(&lock);
  return count;
}

// 1 once a message has been queued or a mailbox closed since `seen` was
// read, 0 if the deadline passes first.
extern "C" std::int64_t merk_mailbox_wait_arrival(
    std::int64_t seen, std::int64_t deadline) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t result{1};
  while (arrivals == seen) {
    if (!WaitArrival(deadline)) {
      result = 0;
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

// The message at `position`, which merk_mailbox_wait has found there.
extern "C" Word merk_mailbox_message(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  pthread_mutex_lock(&lock);
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox || mailbox->count <= position) {
    Fail("a message was read that is not in its mailbox", "");
  }
  Word value{At(mailbox, position)->value};
  pthread_mutex_unlock(&lock);
  return value;
}

extern "C" void merk_mailbox_remove(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(MERK_HANDLE)};
      mailbox && mailbox->count > position) {
    Message **link{&mailbox->head};
    for (std::int64_t i{0}; i < position; ++i) {
      link = &(*link)->next;
    }
    Message *removed{*link};
    *link = removed->next;
    if (mailbox->tail == &removed->next) {
      mailbox->tail = link;
    }
    --mailbox->count;
  }
  pthread_mutex_unlock(&lock);
}

// Registers a mailbox of this node's host under a name, as Erlang's
// register/2 does, which fails if the name is taken or the mailbox has one
// already: 0 if it is registered, 1 if the name is taken, 2 if the mailbox
// has a name, 3 if it is closed or another node's or place's.
extern "C" std::int64_t merk_mailbox_register(
    MERK_HANDLE_PARAMETERS, const char *name) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t result{0};
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox) {
    result = 3;
  } else if (mailbox->name) {
    result = 2;
  } else {
    for (std::int64_t i{0}; i < slotCount; ++i) {
      if (slots[i].open && slots[i].name &&
          std::strcmp(slots[i].name, name) == 0) {
        result = 1;
        break;
      }
    }
  }
  if (result == 0) {
    std::size_t bytes{std::strlen(name) + 1};
    mailbox->name = static_cast<char *>(Malloc(bytes));
    std::memcpy(mailbox->name, name, bytes);
  }
  pthread_mutex_unlock(&lock);
  return result;
}

// Removes a name, if it is registered.
extern "C" void merk_mailbox_unregister(const char *name) {
  pthread_mutex_lock(&lock);
  for (std::int64_t i{0}; i < slotCount; ++i) {
    if (slots[i].open && slots[i].name &&
        std::strcmp(slots[i].name, name) == 0) {
      Unregister(&slots[i]);
      break;
    }
  }
  pthread_mutex_unlock(&lock);
}

extern "C" std::int32_t merk_compare_type_infos(Word x, Word y);

// The mailbox registered under a name on this node: 0 and its slot,
// generation and secret if its message type is the type_info's, 1 if no
// mailbox has the name, 2 if its message type is another.
extern "C" std::int64_t merk_mailbox_lookup(const char *name, Word typeInfo,
    std::int64_t *placeAndSlot, std::int64_t *generation,
    std::int64_t *secret) {
  pthread_mutex_lock(&lock);
  std::int64_t result{1};
  for (std::int64_t i{0}; i < slotCount; ++i) {
    Mailbox &mailbox{slots[i]};
    if (mailbox.open && mailbox.name && std::strcmp(mailbox.name, name) == 0) {
      result =
          merk_compare_type_infos(mailbox.messageType, typeInfo) == 0 ? 0 : 2;
      if (result == 0) {
        *placeAndSlot = i;
        *generation = mailbox.generation;
        *secret = static_cast<std::int64_t>(mailbox.secret);
      }
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}
