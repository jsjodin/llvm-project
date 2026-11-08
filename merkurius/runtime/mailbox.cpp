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
#include "device-ring.h"
#include "fragment.h"
#include "node.h"
#include "process.h"
#include "walk.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sched.h>
#include <sys/random.h>

namespace {

using namespace merkurius::runtime;

[[noreturn]] void Fail(const char *message, const char *detail) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%s\n", message, detail);
  std::exit(1);
}

Word CopyTyped(Word value, const Type &type, Fragment &fragment);

// Whether a value of the type is its word in a copy.
bool CopiedAsItIs(const TypeCtor *ctor) {
  switch (ctor->kind) {
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
    return true;
  default:
    return false;
  }
}

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
    // What is copied as it is, as CopyTyped returns it, is copied as a block.
    if (size != 0 && CopiedAsItIs(Resolve(element, &type.environment).ctor)) {
      std::memcpy(copy + 1, array + 1, sizeof(Word) * size);
      return reinterpret_cast<Word>(copy);
    }
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
  // The ring through which a device receives from it, or null, and whether
  // its messages are arrays; of which the first `consumed` were taken, and
  // have gone from the list.
  DeviceRing *ring;
  int ringDevice;
  bool ringOfArrays;
  std::uint64_t consumed;
  // The process that receives from it, which a message wakes, or null if
  // made on a thread that runs none; and its neighbours in that process's
  // list of mailboxes, as slots, or -1.
  Process *owner;
  std::int64_t previousOwned;
  std::int64_t nextOwned;
};

pthread_mutex_t &lock{schedulerLock};
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
// How many mailboxes have rings, so that most regions' ends need no walk.
std::int64_t attachedRings{0};

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
// Adds a mailbox to its owner's list; a null owner has none.
void Own(std::int64_t index, Process *owner) {
  Mailbox &mailbox{slots[index]};
  mailbox.owner = owner;
  mailbox.previousOwned = -1;
  mailbox.nextOwned = -1;
  if (owner) {
    mailbox.nextOwned = owner->firstMailbox;
    if (owner->firstMailbox >= 0) {
      slots[owner->firstMailbox].previousOwned = index;
    }
    owner->firstMailbox = index;
  }
}

void Disown(Mailbox *mailbox) {
  Process *owner{mailbox->owner};
  if (!owner) {
    return;
  }
  if (mailbox->previousOwned >= 0) {
    slots[mailbox->previousOwned].nextOwned = mailbox->nextOwned;
  } else {
    owner->firstMailbox = mailbox->nextOwned;
  }
  if (mailbox->nextOwned >= 0) {
    slots[mailbox->nextOwned].previousOwned = mailbox->previousOwned;
  }
  mailbox->owner = nullptr;
}

// Only its owner receives from a mailbox. A thread that runs no process, as
// on the host's side of a device, is not one; a mailbox made on such a thread
// goes to the first process that receives from it.
void CheckOwner(Mailbox *mailbox) {
  Process *process{CurrentProcess()};
  if (!process) {
    return;
  }
  if (!mailbox->owner) {
    Own(mailbox - slots, process);
  } else if (mailbox->owner != process) {
    Fail("a process received from a mailbox of another process", "");
  }
}

void Unregister(Mailbox *mailbox) {
  std::free(mailbox->name);
  mailbox->name = nullptr;
}

// Drops the messages the device has taken from the list. Called with the
// lock held.
void TrimRing(Mailbox *mailbox) {
  std::uint64_t head{__atomic_load_n(&mailbox->ring->head, __ATOMIC_ACQUIRE)};
  for (; mailbox->consumed != head; ++mailbox->consumed) {
    Message *taken{mailbox->head};
    mailbox->head = taken->next;
    if (!mailbox->head) {
      mailbox->tail = &mailbox->head;
    }
    --mailbox->count;
  }
}

// Shows the device as many of the list's messages as fit, and tells it if
// any are left. Called with the lock held.
void FillRing(Mailbox *mailbox) {
  TrimRing(mailbox);
  DeviceRing *ring{mailbox->ring};
  std::uint64_t tail{ring->tail};
  std::uint64_t shown{tail - mailbox->consumed};
  Message *message{mailbox->head};
  for (std::uint64_t i{0}; i < shown; ++i) {
    message = message->next;
  }
  while (message && tail - mailbox->consumed < kRingSlots) {
    ring->slots[tail % kRingSlots] = mailbox->ringOfArrays
        ? *reinterpret_cast<const Word *>(message->value)
        : message->value;
    ++tail;
    __atomic_store_n(&ring->tail, tail, __ATOMIC_RELEASE);
    message = message->next;
  }
  __atomic_store_n(&ring->spilled, message != nullptr, __ATOMIC_RELEASE);
}

// Queues a message whose cells are in a fragment of their own. Called with
// the lock held.
void Queue(Mailbox *mailbox, Message *message) {
  message->stamp = ++stamps;
  *mailbox->tail = message;
  mailbox->tail = &message->next;
  ++mailbox->count;
  __atomic_add_fetch(&arrivals, 1, __ATOMIC_RELEASE);
  if (mailbox->ring) {
    FillRing(mailbox);
  }
  pthread_cond_broadcast(&arrived);
  if (mailbox->owner) {
    Wake(mailbox->owner);
  }
}

// Waits on `arrived` until the deadline, or for ever if it is negative:
// false once the deadline has passed. Called with the lock held.
bool WaitArrival(std::int64_t deadline) {
  if (CanSuspend()) {
    // A process lets others run; the scheduler spins when nothing can.
    if (deadline >= 0 && Now() >= deadline) {
      return false;
    }
    Block(deadline);
    return true;
  }
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

// When a wait that ends at the deadline may stop spinning: never past the
// deadline, so that a receive with no time to wait does not spin.
std::int64_t SpinEnd(std::int64_t deadline) {
  if (CanSuspend()) {
    return 0;
  }
  std::int64_t end{Now() + SpinNanoseconds()};
  return deadline >= 0 && deadline < end ? deadline : end;
}

// Spins, without the lock, until something has arrived or the time is
// `end`. Called with the lock held, which it holds again on return.
void SpinForArrival(std::int64_t end) {
  std::int64_t seen{arrivals};
  pthread_mutex_unlock(&lock);
  // It yields, so that with fewer cores than threads the one it waits for
  // can run.
  while (__atomic_load_n(&arrivals, __ATOMIC_ACQUIRE) == seen && Now() < end) {
    sched_yield();
  }
  pthread_mutex_lock(&lock);
}

pthread_once_t spinOnce = PTHREAD_ONCE_INIT;
std::int64_t spinNanoseconds{50000};

void ReadSpin() {
  if (const char *microseconds{std::getenv("MERK_SPIN_MICROSECONDS")}) {
    spinNanoseconds = std::strtoll(microseconds, nullptr, 10) * 1000;
  }
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

std::int64_t SpinNanoseconds() {
  pthread_once(&spinOnce, ReadSpin);
  return spinNanoseconds;
}

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
  mailbox.ring = nullptr;
  mailbox.ringDevice = -1;
  mailbox.ringOfArrays = false;
  mailbox.consumed = 0;
  Own(index, CurrentProcess());
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
// receive waiting on it wakes and finds nothing. Called with the lock held.
static void Close(Mailbox *mailbox) {
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
  // Whoever made the ring frees it when no region is running.
  mailbox->ring = nullptr;
  __atomic_add_fetch(&arrivals, 1, __ATOMIC_RELEASE);
  pthread_cond_broadcast(&arrived);
  if (Process *owner{mailbox->owner}) {
    Disown(mailbox);
    Wake(owner);
  }
}

extern "C" void merk_mailbox_close(MERK_HANDLE_PARAMETERS) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    Close(mailbox);
  }
  pthread_mutex_unlock(&lock);
}

namespace merkurius::runtime {
void CloseMailboxesOf(Process *process) {
  pthread_once(&once, InitializeCondition);
  while (process->firstMailbox >= 0) {
    Close(&slots[process->firstMailbox]);
  }
}
} // namespace merkurius::runtime

// A new process runs `closure` on `argument`, the handle's mailbox, which
// it now owns; `stackBytes` of 0 is the default stack. Temporary (chunk 2,
// docs/INPROGRESS.md, The scheduler, before the collector): the closure and
// the handle's cell are shared with the parent, not copied, as a closure's
// cell does not say the types of what it captured.
extern "C" void merk_process_spawn(std::int64_t stackBytes, Word closure,
    Word argument, MERK_HANDLE_PARAMETERS) {
  if (!CurrentProcess()) {
    Fail("spawn is not available on a thread that runs no process", "");
  }
  pthread_mutex_lock(&lock);
  Process *child{
      NewProcess(static_cast<std::size_t>(stackBytes), closure, argument)};
  if (!child) {
    Fail("no address space for a process's stack", "");
  }
  Mailbox *mailbox{Find(MERK_HANDLE)};
  Disown(mailbox);
  Own(mailbox - slots, child);
  Start(child);
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
  std::int64_t spinEnd{SpinEnd(deadline)};
  pthread_mutex_lock(&lock);
  std::int64_t result{0};
  for (;;) {
    Mailbox *mailbox{Find(MERK_HANDLE)};
    if (!mailbox) {
      break;
    }
    CheckOwner(mailbox);
    if (mailbox->count > position) {
      result = 1;
      break;
    }
    if (Now() < spinEnd) {
      SpinForArrival(spinEnd);
      continue;
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
  if (mailbox) {
    CheckOwner(mailbox);
  }
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

// The stamp of the last message queued, which a receive over a selector
// reads before it looks at its mailboxes: every message with a stamp up to
// it was queued then, so the oldest of those it finds is the oldest of all,
// while one it finds with a later stamp arrived as it looked, and an older
// one may have arrived too, in a mailbox it had looked at already.
extern "C" std::int64_t merk_mailbox_last_stamp(void) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t stamp{static_cast<std::int64_t>(stamps)};
  pthread_mutex_unlock(&lock);
  return stamp;
}

// 1 once a message has been queued or a mailbox closed since `seen` was
// read, 0 if the deadline passes first.
extern "C" std::int64_t merk_mailbox_wait_arrival(
    std::int64_t seen, std::int64_t deadline) {
  pthread_once(&once, InitializeCondition);
  std::int64_t spinEnd{SpinEnd(deadline)};
  pthread_mutex_lock(&lock);
  std::int64_t result{1};
  while (arrivals == seen) {
    if (Now() < spinEnd) {
      SpinForArrival(spinEnd);
      continue;
    }
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

// The mailbox's ring for a device (device-ring.h), made by `allocate` the
// first time, and its head; null if the mailbox is closed or not this
// node's host's. What it holds already is shown in the ring.
extern "C" DeviceRing *merk_mailbox_attach(MERK_HANDLE_PARAMETERS, bool arrays,
    int device, DeviceRing *(*allocate)(int), std::uint64_t *head) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  DeviceRing *ring{nullptr};
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    if (!mailbox->ring) {
      mailbox->ring = allocate(device);
      mailbox->ringDevice = device;
      mailbox->ringOfArrays = arrays;
      mailbox->consumed = 0;
      ++attachedRings;
      FillRing(mailbox);
    } else if (mailbox->ringDevice != device) {
      // A ring has one consumer. This runs on the plugin's server thread,
      // where exit would hang.
      std::fflush(stdout);
      std::fprintf(stderr,
          "Mercury runtime: two devices receive from one mailbox, which is "
          "not supported yet\n");
      std::_Exit(1);
    }
    ring = mailbox->ring;
    *head = ring->head;
  }
  pthread_mutex_unlock(&lock);
  return ring;
}

// Shows the device the messages that did not fit in its ring.
extern "C" void merk_mailbox_drain(MERK_HANDLE_PARAMETERS) {
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(MERK_HANDLE)}; mailbox && mailbox->ring) {
    FillRing(mailbox);
  }
  pthread_mutex_unlock(&lock);
}

// Drops from the list what the device has taken, so that the mailbox's
// oldest message is the next it will take.
extern "C" void merk_mailbox_trim(MERK_HANDLE_PARAMETERS) {
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(MERK_HANDLE)}; mailbox && mailbox->ring) {
    TrimRing(mailbox);
  }
  pthread_mutex_unlock(&lock);
}

// When no region is running: every mailbox goes back to its list alone,
// which still has what the devices did not take. The caller frees the rings.
extern "C" void merk_mailbox_detach_rings(void) {
  pthread_mutex_lock(&lock);
  if (attachedRings != 0) {
    for (std::int64_t i{0}; i < slotCount; ++i) {
      if (slots[i].open && slots[i].ring) {
        TrimRing(&slots[i]);
        slots[i].ring = nullptr;
      }
    }
    attachedRings = 0;
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
