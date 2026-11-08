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

#include <atomic>
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

// A type_info: a descriptor is static, and a cell of one and its arguments'
// type_infos is copied.
Word CopyTypeInfo(Word info, Fragment &fragment) {
  if ((info & kTagMask) == 0) {
    return info;
  }
  const Word *cell{reinterpret_cast<const Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word arity{counted ? cell[1] : ctor->arity};
  Word before{counted ? Word{2} : Word{1}};
  auto *copy{
      static_cast<Word *>(fragment.Allocate(sizeof(Word) * (before + arity)))};
  for (Word i{0}; i < before; ++i) {
    copy[i] = cell[i];
  }
  for (Word i{0}; i < arity; ++i) {
    copy[before + i] = CopyTypeInfo(cell[before + i], fragment);
  }
  return reinterpret_cast<Word>(copy) + 1;
}

// A typeclass_info, by its method table's shape. The table is static.
Word CopyDictionary(Word dictionary, Fragment &fragment) {
  const Word *cell{reinterpret_cast<const Word *>(dictionary)};
  DictionaryShape shape{ShapeOfDictionary(cell)};
  Word total{shape.superclasses + shape.typeInfos + shape.constraints};
  auto *copy{
      static_cast<Word *>(fragment.Allocate(sizeof(Word) * (1 + total)))};
  copy[0] = cell[0];
  for (Word i{1}; i <= total; ++i) {
    bool isTypeInfo{i > shape.superclasses &&
        i <= shape.superclasses + shape.typeInfos};
    copy[i] = isTypeInfo ? CopyTypeInfo(cell[i], fragment)
                         : CopyDictionary(cell[i], fragment);
  }
  return reinterpret_cast<Word>(copy);
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
    unsigned tag{type.ctor->kind == Tuple
            ? functor->tag
            : static_cast<unsigned>(value & kTagMask)};
    // A constructor that shares its tag has its secondary tag in word 0.
    const Word *from{reinterpret_cast<const Word *>(value - tag)};
    unsigned before{static_cast<unsigned>(fields - from)};
    unsigned arity{functor->arity};
    Word *cell{static_cast<Word *>(
        fragment.Allocate(sizeof(Word) * (before + arity)))};
    // Its secondary tag, and then the type_infos and typeclass_infos of an
    // existentially typed constructor, which the fields' types read.
    unsigned hidden{functor->hidden};
    for (unsigned i{0}; i < before - hidden; ++i) {
      cell[i] = from[i];
    }
    const Word *hiddenFrom{fields - hidden};
    Word *hiddenTo{cell + before - hidden};
    for (unsigned i{0}; i < hidden; ++i) {
      hiddenTo[i] = i < functor->hiddenTypeInfos
          ? CopyTypeInfo(hiddenFrom[i], fragment)
          : CopyDictionary(hiddenFrom[i], fragment);
    }
    Environment existential{type.environment.arguments,
        type.environment.parent, hiddenFrom};
    const Environment *environment{
        hidden != 0 ? &existential : &type.environment};
    *slot = reinterpret_cast<Word>(cell) + tag;
    Word *copied{cell + before};
    if (arity == 0) {
      return result;
    }
    for (unsigned i{0}; i + 1 < arity; ++i) {
      copied[i] =
          CopyValue(fields[i], functor->fields[i], environment, fragment);
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      copied[arity - 1] =
          CopyValue(fields[arity - 1], last, environment, fragment);
      return result;
    }
    value = fields[arity - 1];
    slot = &copied[arity - 1];
  }
}

// A closure's cell, by its layout: its code is static, and each capture is
// copied by what the layout says. A capture's type is found in the original
// cell, whose captured type_infos are the ones it names.
Word CopyClosure(Word value, Fragment &fragment) {
  const Word *cell{reinterpret_cast<const Word *>(value)};
  const auto *layout{reinterpret_cast<const ClosureLayout *>(cell[1])};
  if (!layout) {
    Fail("cannot copy a closure made on a device", "");
  }
  Word *copy{static_cast<Word *>(
      fragment.Allocate(sizeof(Word) * (2 + layout->count)))};
  copy[0] = cell[0];
  copy[1] = cell[1];
  const Word *captured{cell + 2};
  Environment environment{nullptr, nullptr, captured};
  for (Word i{0}; i < layout->count; ++i) {
    Word capture{layout->captures[i]};
    if (capture == kCaptureUnknown) {
      Fail("cannot copy a closure that captured a value of a type with no "
           "description", "");
    }
    if (capture == kCaptureTypeInfo) {
      copy[2 + i] = CopyTypeInfo(captured[i], fragment);
    } else if (capture == kCaptureTypeclassInfo) {
      copy[2 + i] = CopyDictionary(captured[i], fragment);
    } else {
      copy[2 + i] = CopyValue(captured[i], capture, &environment, fragment);
    }
  }
  return reinterpret_cast<Word>(copy);
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
    return CopyClosure(value, fragment);
  case Void:
    // Nothing has a value of this type.
    std::abort();
  }
  Fail("cannot send a value of type ", type.ctor->name);
}

// What a process that ended left for its monitor's owner, whose function
// makes the message from it on the receiver's side (Monitors, below).
struct PendingDown {
  std::int64_t monitor;
  std::int64_t kind;
  const char *text;
};

struct Message {
  Message *next;
  // With `down`, the monitor's function.
  Word value;
  PendingDown *down;
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
  std::int64_t index;
  // Guards the rest, but for its neighbours in the list, which its owner's
  // waitLock guards, and its name, which registryLock does. It lives as long
  // as the slot.
  pthread_mutex_t lock;
};

// Guards the free slots, new chunks of slots, and names.
pthread_mutex_t registryLock = PTHREAD_MUTEX_INITIALIZER;
// Where a thread that cannot suspend a process waits for a message: one that
// runs no process, or a process in a target region.
pthread_mutex_t waitersLock = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t nodeOnce = PTHREAD_ONCE_INIT;
Word nodeHigh{0};
Word nodeLow{0};

void ChooseNode() {
  nodeHigh = Random();
  nodeLow = Random();
}
pthread_cond_t arrived;
pthread_once_t once = PTHREAD_ONCE_INIT;
// The counts of mailboxes that no process owns, and of threads that run none.
ArrivalCounts unowned{};
// How many threads are about to wait on `arrived`, and a count that tells
// them something happened since they looked.
std::int64_t waiters{0};
std::int64_t waitersEvents{0};

// Slots are in chunks that never move, so that a slot's lock and queue stay
// put and finding one needs no lock.
constexpr unsigned kChunkShift{10};
constexpr std::int64_t kChunkSlots{1 << kChunkShift};
constexpr std::int64_t kMostChunks{1 << 20};
Mailbox *chunks[kMostChunks];
std::int64_t slotCount{0};
std::int64_t firstFree{-1};
// How many mailboxes have rings, so that most regions' ends need no walk.
std::int64_t attachedRings{0};

Mailbox *SlotAt(std::int64_t index) {
  return &__atomic_load_n(&chunks[index >> kChunkShift],
      __ATOMIC_ACQUIRE)[index & (kChunkSlots - 1)];
}

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

// The mailbox of a handle, with its lock held, or null for a closed one, or
// one of another node or place.
Mailbox *Find(const Handle &handle) {
  pthread_once(&nodeOnce, ChooseNode);
  if (handle.nodeHigh != nodeHigh || handle.nodeLow != nodeLow ||
      handle.placeAndSlot >> kPlaceShift != 0) {
    return nullptr;
  }
  Word slot{handle.placeAndSlot & kSlotMask};
  if (slot >= static_cast<Word>(__atomic_load_n(&slotCount, __ATOMIC_ACQUIRE))) {
    return nullptr;
  }
  Mailbox *mailbox{SlotAt(static_cast<std::int64_t>(slot))};
  pthread_mutex_lock(&mailbox->lock);
  if (mailbox->open && mailbox->generation == handle.generation &&
      mailbox->secret == handle.secret) {
    return mailbox;
  }
  pthread_mutex_unlock(&mailbox->lock);
  return nullptr;
}

void Unlock(Mailbox *mailbox) { pthread_mutex_unlock(&mailbox->lock); }

ArrivalCounts &CountsOf(const Mailbox *mailbox) {
  return mailbox->owner ? mailbox->owner->counts : unowned;
}

ArrivalCounts &MyCounts() {
  Process *process{CurrentProcess()};
  return process ? process->counts : unowned;
}

// Adds a mailbox, with its lock held, to its owner's list; a null owner has
// none.
void Own(Mailbox *mailbox, Process *owner) {
  mailbox->owner = owner;
  mailbox->previousOwned = -1;
  mailbox->nextOwned = -1;
  if (owner) {
    pthread_mutex_lock(&owner->waitLock);
    mailbox->nextOwned = owner->firstMailbox;
    if (owner->firstMailbox >= 0) {
      SlotAt(owner->firstMailbox)->previousOwned = mailbox->index;
    }
    owner->firstMailbox = mailbox->index;
    pthread_mutex_unlock(&owner->waitLock);
  }
}

// With the mailbox's lock and its owner's waitLock held.
void Unlink(Mailbox *mailbox) {
  Process *owner{mailbox->owner};
  if (mailbox->previousOwned >= 0) {
    SlotAt(mailbox->previousOwned)->nextOwned = mailbox->nextOwned;
  } else {
    owner->firstMailbox = mailbox->nextOwned;
  }
  if (mailbox->nextOwned >= 0) {
    SlotAt(mailbox->nextOwned)->previousOwned = mailbox->previousOwned;
  }
  mailbox->owner = nullptr;
}

void Disown(Mailbox *mailbox) {
  if (Process *owner{mailbox->owner}) {
    pthread_mutex_lock(&owner->waitLock);
    Unlink(mailbox);
    pthread_mutex_unlock(&owner->waitLock);
  }
}

// Its queued messages get new stamps, from the counts it now uses, so that
// they compare with the receiver's others. With its lock held.
void Restamp(Mailbox *mailbox) {
  ArrivalCounts &counts{CountsOf(mailbox)};
  for (Message *message{mailbox->head}; message; message = message->next) {
    message->stamp = __atomic_add_fetch(&counts.stamps, 1, __ATOMIC_SEQ_CST);
  }
}

// Only its owner receives from a mailbox. A thread that runs no process, as
// on the host's side of a device, has no counts of its own: a mailbox it
// receives from goes to no process, so that it and the senders use the same
// ones. A mailbox of no process goes to the first process that receives from
// it. With its lock held.
void CheckOwner(Mailbox *mailbox) {
  Process *process{CurrentProcess()};
  if (!process) {
    if (mailbox->owner) {
      Disown(mailbox);
      Restamp(mailbox);
    }
    return;
  }
  if (!mailbox->owner) {
    Own(mailbox, process);
    Restamp(mailbox);
  } else if (mailbox->owner != process) {
    Fail("a process received from a mailbox of another process", "");
  }
}

// With registryLock held.
void RemoveName(Mailbox *mailbox) {
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
    if (message->down) {
      // This can be on the plugin's server thread, where exit would hang.
      std::fflush(stdout);
      std::fprintf(stderr,
          "Mercury runtime: a monitor's message to a mailbox a device "
          "receives from is not supported yet\n");
      std::_Exit(1);
    }
    ring->slots[tail % kRingSlots] = mailbox->ringOfArrays
        ? *reinterpret_cast<const Word *>(message->value)
        : message->value;
    ++tail;
    __atomic_store_n(&ring->tail, tail, __ATOMIC_RELEASE);
    message = message->next;
  }
  __atomic_store_n(&ring->spilled, message != nullptr, __ATOMIC_RELEASE);
}

// Something a thread waiting on `arrived` may be waiting for has happened.
void TellWaiters() {
  if (__atomic_load_n(&waiters, __ATOMIC_SEQ_CST) > 0) {
    pthread_mutex_lock(&waitersLock);
    __atomic_add_fetch(&waitersEvents, 1, __ATOMIC_SEQ_CST);
    pthread_cond_broadcast(&arrived);
    pthread_mutex_unlock(&waitersLock);
  }
}

// Queues a message whose cells are in a fragment of their own, with the
// mailbox's lock held. The count goes up before the owner's waitLock is
// taken, so that a receive about to block either sees it or is woken.
void Queue(Mailbox *mailbox, Message *message) {
  ArrivalCounts &counts{CountsOf(mailbox)};
  message->stamp = __atomic_add_fetch(&counts.stamps, 1, __ATOMIC_SEQ_CST);
  *mailbox->tail = message;
  mailbox->tail = &message->next;
  ++mailbox->count;
  __atomic_add_fetch(&counts.arrivals, 1, __ATOMIC_SEQ_CST);
  if (mailbox->ring) {
    FillRing(mailbox);
  }
  if (Process *owner{mailbox->owner}) {
    pthread_mutex_lock(&owner->waitLock);
    Wake(owner);
    pthread_mutex_unlock(&owner->waitLock);
  }
  TellWaiters();
}

// A thread that cannot suspend a process is about to look for what it waits
// for: anything that happens after this reaches WaitForEvent.
std::int64_t StartLooking() {
  __atomic_add_fetch(&waiters, 1, __ATOMIC_SEQ_CST);
  return __atomic_load_n(&waitersEvents, __ATOMIC_SEQ_CST);
}

void StopLooking() { __atomic_sub_fetch(&waiters, 1, __ATOMIC_SEQ_CST); }

// Waits for an event after `seen` until the deadline, or for ever if it is
// negative: false once the deadline has passed.
bool WaitForEvent(std::int64_t seen, std::int64_t deadline) {
  if (deadline >= 0 && Now() >= deadline) {
    return false;
  }
  pthread_mutex_lock(&waitersLock);
  if (__atomic_load_n(&waitersEvents, __ATOMIC_SEQ_CST) == seen) {
    if (deadline < 0) {
      pthread_cond_wait(&arrived, &waitersLock);
    } else {
      timespec until{static_cast<time_t>(deadline / 1000000000),
          static_cast<long>(deadline % 1000000000)};
      pthread_cond_timedwait(&arrived, &waitersLock, &until);
    }
  }
  pthread_mutex_unlock(&waitersLock);
  return true;
}

// When a wait that ends at the deadline may stop spinning: never past the
// deadline, so that a receive with no time to wait does not spin.
std::int64_t SpinEnd(std::int64_t deadline) {
  std::int64_t end{Now() + SpinNanoseconds()};
  return deadline >= 0 && deadline < end ? deadline : end;
}

// Spins until the number of arrivals is no longer `seen`, or the time is
// `end`. It yields, so that with fewer cores than threads the one it waits
// for can run.
void SpinForArrival(std::int64_t seen, std::int64_t end) {
  ArrivalCounts &counts{MyCounts()};
  while (__atomic_load_n(&counts.arrivals, __ATOMIC_ACQUIRE) == seen &&
      Now() < end) {
    sched_yield();
  }
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
  // Decoded with the mailbox's lock held, so that it, and its type, stay.
  Delivery delivery{Delivery::Dropped};
  if (Mailbox *mailbox{Find(handle)}) {
    Fragment fragment;
    auto *message{static_cast<Message *>(fragment.Allocate(sizeof(Message)))};
    message->next = nullptr;
    message->down = nullptr;
    if (Decode(bytes, size, mailbox->messageType, fragment, message->value)) {
      delivery = Delivery::Undecodable;
    } else {
      Queue(mailbox, message);
      delivery = Delivery::Delivered;
    }
    Unlock(mailbox);
  }
  return delivery;
}

// A mailbox with a name, whose fields do not change while registryLock is
// held, as closing it takes the name away under that lock first.
static Mailbox *Named(const char *name) {
  std::int64_t count{__atomic_load_n(&slotCount, __ATOMIC_ACQUIRE)};
  for (std::int64_t i{0}; i < count; ++i) {
    Mailbox *mailbox{SlotAt(i)};
    if (mailbox->name && std::strcmp(mailbox->name, name) == 0) {
      return mailbox;
    }
  }
  return nullptr;
}

std::int64_t LookupForPeer(const char *name,
    const std::uint8_t fingerprint[kFingerprintBytes], Handle &found) {
  pthread_mutex_lock(&registryLock);
  std::int64_t result{1};
  if (Mailbox *mailbox{Named(name)}) {
    std::uint8_t own[kFingerprintBytes];
    Fingerprint(mailbox->messageType, own);
    result = std::memcmp(own, fingerprint, kFingerprintBytes) == 0 ? 0 : 2;
    if (result == 0) {
      found = {nodeHigh, nodeLow, static_cast<Word>(mailbox->index),
          mailbox->generation, mailbox->secret};
    }
  }
  pthread_mutex_unlock(&registryLock);
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
  pthread_mutex_lock(&registryLock);
  std::int64_t index{firstFree};
  if (index >= 0) {
    firstFree = SlotAt(index)->nextFree;
  } else {
    index = slotCount;
    if ((index & (kChunkSlots - 1)) == 0) {
      if (index >> kChunkShift == kMostChunks) {
        Fail("too many mailboxes", "");
      }
      auto *chunk{static_cast<Mailbox *>(
          std::calloc(kChunkSlots, sizeof(Mailbox)))};
      if (!chunk) {
        Fail("out of memory", "");
      }
      for (std::int64_t i{0}; i < kChunkSlots; ++i) {
        pthread_mutex_init(&chunk[i].lock, nullptr);
        chunk[i].index = index + i;
      }
      __atomic_store_n(&chunks[index >> kChunkShift], chunk, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&slotCount, index + 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&registryLock);
  Mailbox &mailbox{*SlotAt(index)};
  pthread_mutex_lock(&mailbox.lock);
  mailbox.secret = Random();
  mailbox.messageType = KeepTypeInfo(typeInfo);
  mailbox.head = nullptr;
  mailbox.tail = &mailbox.head;
  mailbox.count = 0;
  mailbox.nextFree = -1;
  mailbox.ring = nullptr;
  mailbox.ringDevice = -1;
  mailbox.ringOfArrays = false;
  mailbox.consumed = 0;
  Own(&mailbox, CurrentProcess());
  mailbox.open = true;
  *placeAndSlot = index;
  *generation = mailbox.generation;
  *secret = static_cast<std::int64_t>(mailbox.secret);
  pthread_mutex_unlock(&mailbox.lock);
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
// receive waiting on it wakes and finds nothing. With the mailbox's lock held.
static void Close(Mailbox *mailbox) {
  pthread_mutex_lock(&registryLock);
  if (mailbox->name) {
    RemoveName(mailbox);
  }
  pthread_mutex_unlock(&registryLock);
  FreeTypeInfo(mailbox->messageType);
  mailbox->messageType = 0;
  mailbox->open = false;
  ++mailbox->generation;
  mailbox->head = nullptr;
  mailbox->tail = &mailbox->head;
  mailbox->count = 0;
  // Whoever made the ring frees it when no region is running.
  mailbox->ring = nullptr;
  __atomic_add_fetch(&CountsOf(mailbox).arrivals, 1, __ATOMIC_SEQ_CST);
  if (Process *owner{mailbox->owner}) {
    pthread_mutex_lock(&owner->waitLock);
    Unlink(mailbox);
    Wake(owner);
    pthread_mutex_unlock(&owner->waitLock);
  }
  TellWaiters();
  // Last, so that whoever takes the slot next waits for this one's lock.
  pthread_mutex_lock(&registryLock);
  mailbox->nextFree = firstFree;
  firstFree = mailbox->index;
  pthread_mutex_unlock(&registryLock);
}

extern "C" void merk_mailbox_close(MERK_HANDLE_PARAMETERS) {
  pthread_once(&once, InitializeCondition);
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    Close(mailbox);
    Unlock(mailbox);
  }
}

namespace merkurius::runtime {
void CloseMailboxesOf(Process *process) {
  pthread_once(&once, InitializeCondition);
  for (;;) {
    pthread_mutex_lock(&process->waitLock);
    std::int64_t first{process->firstMailbox};
    pthread_mutex_unlock(&process->waitLock);
    if (first < 0) {
      return;
    }
    Mailbox *mailbox{SlotAt(first)};
    pthread_mutex_lock(&mailbox->lock);
    if (mailbox->owner == process) {
      Close(mailbox);
    }
    pthread_mutex_unlock(&mailbox->lock);
  }
}
} // namespace merkurius::runtime

// A new process runs `closure` on `argument`, the handle's mailbox, which
// it now owns; `stackBytes` of 0 is the default stack. Temporary (chunk 2,
// docs/INPROGRESS.md, Before the collector): the closure and
// the handle's cell are shared with the parent, not copied, as a closure's
// cell does not say the types of what it captured.
extern "C" void merk_process_spawn(std::int64_t stackBytes, Word closure,
    Word argument, MERK_HANDLE_PARAMETERS) {
  if (!CurrentProcess()) {
    Fail("spawn is not available on a thread that runs no process", "");
  }
  pthread_mutex_lock(&schedulerLock);
  Process *child{
      NewProcess(static_cast<std::size_t>(stackBytes), closure, argument)};
  pthread_mutex_unlock(&schedulerLock);
  if (!child) {
    Fail("no address space for a process's stack", "");
  }
  Mailbox *mailbox{Find(MERK_HANDLE)};
  Disown(mailbox);
  Own(mailbox, child);
  Unlock(mailbox);
  Start(child);
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
  message->down = nullptr;
  message->value = CopyValue(value, typeInfo, nullptr, fragment);
  if (Mailbox *mailbox{Find(handle)}) {
    Queue(mailbox, message);
    Unlock(mailbox);
  }
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
  bool suspends{CanSuspend()};
  std::int64_t spinEnd{suspends ? 0 : SpinEnd(deadline)};
  std::int64_t result{0};
  for (;;) {
    std::int64_t seen{suspends ? 0 : StartLooking()};
    std::int64_t arrivals{
        __atomic_load_n(&MyCounts().arrivals, __ATOMIC_SEQ_CST)};
    Mailbox *mailbox{Find(MERK_HANDLE)};
    if (!mailbox) {
      if (!suspends) {
        StopLooking();
      }
      break;
    }
    CheckOwner(mailbox);
    if (mailbox->count > position) {
      Unlock(mailbox);
      if (!suspends) {
        StopLooking();
      }
      result = 1;
      break;
    }
    if (suspends) {
      // The process scheduled next spins if nothing can run.
      if (deadline >= 0 && Now() >= deadline) {
        Unlock(mailbox);
        break;
      }
      pthread_mutex_lock(&CurrentProcess()->waitLock);
      Unlock(mailbox);
      Block(deadline);
      continue;
    }
    Unlock(mailbox);
    bool waited{true};
    if (Now() < spinEnd) {
      SpinForArrival(arrivals, spinEnd);
    } else {
      waited = WaitForEvent(seen, deadline);
    }
    StopLooking();
    if (!waited) {
      break;
    }
  }
  return result;
}

// For a receive over a selector, without waiting: the stamp of the
// message at `position`, which is positive, 0 if the mailbox has no message
// there yet, or -1 if it is closed or not this node's host's.
extern "C" std::int64_t merk_mailbox_poll(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  pthread_once(&once, InitializeCondition);
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox) {
    return -1;
  }
  CheckOwner(mailbox);
  std::int64_t result{mailbox->count > position
          ? static_cast<std::int64_t>(At(mailbox, position)->stamp)
          : 0};
  Unlock(mailbox);
  return result;
}

// The number of messages queued and mailboxes closed so far, which a
// receive over a selector reads before it looks at its mailboxes.
extern "C" std::int64_t merk_mailbox_arrivals(void) {
  pthread_once(&once, InitializeCondition);
  return __atomic_load_n(&MyCounts().arrivals, __ATOMIC_SEQ_CST);
}

// The stamp of the last message queued, which a receive over a selector
// reads before it looks at its mailboxes: every message with a stamp up to
// it was queued then, or is being queued by a sender that holds its mailbox's
// lock, which looking at that mailbox waits for. So the oldest of those it
// finds is the oldest of all, while one it finds with a later stamp arrived
// as it looked, and an older one may have arrived too, in a mailbox it had
// looked at already.
extern "C" std::int64_t merk_mailbox_last_stamp(void) {
  pthread_once(&once, InitializeCondition);
  return static_cast<std::int64_t>(
      __atomic_load_n(&MyCounts().stamps, __ATOMIC_SEQ_CST));
}

// 1 once a message has been queued or a mailbox closed since `seen` was
// read, 0 if the deadline passes first.
extern "C" std::int64_t merk_mailbox_wait_arrival(
    std::int64_t seen, std::int64_t deadline) {
  pthread_once(&once, InitializeCondition);
  ArrivalCounts &counts{MyCounts()};
  if (CanSuspend()) {
    Process *self{CurrentProcess()};
    for (;;) {
      pthread_mutex_lock(&self->waitLock);
      if (__atomic_load_n(&counts.arrivals, __ATOMIC_SEQ_CST) != seen) {
        pthread_mutex_unlock(&self->waitLock);
        return 1;
      }
      if (deadline >= 0 && Now() >= deadline) {
        pthread_mutex_unlock(&self->waitLock);
        return 0;
      }
      Block(deadline);
    }
  }
  std::int64_t spinEnd{SpinEnd(deadline)};
  for (;;) {
    std::int64_t event{StartLooking()};
    if (__atomic_load_n(&counts.arrivals, __ATOMIC_SEQ_CST) != seen) {
      StopLooking();
      return 1;
    }
    bool waited{true};
    if (Now() < spinEnd) {
      SpinForArrival(seen, spinEnd);
    } else {
      waited = WaitForEvent(event, deadline);
    }
    StopLooking();
    if (!waited) {
      return 0;
    }
  }
}

// The message at `position`, which merk_mailbox_wait has found there.
extern "C" Word merk_mailbox_message(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox || mailbox->count <= position) {
    Fail("a message was read that is not in its mailbox", "");
  }
  Message *message{At(mailbox, position)};
  if (message->down) {
    Fail("a monitor's message was read outside a receive", "");
  }
  Word value{message->value};
  Unlock(mailbox);
  return value;
}

extern "C" void merk_mailbox_remove(
    MERK_HANDLE_PARAMETERS, std::int64_t position) {
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox) {
    return;
  }
  if (mailbox->count > position) {
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
  Unlock(mailbox);
}

// The mailbox's ring for a device (device-ring.h), made by `allocate` the
// first time, and its head; null if the mailbox is closed or not this
// node's host's. What it holds already is shown in the ring.
extern "C" DeviceRing *merk_mailbox_attach(MERK_HANDLE_PARAMETERS, bool arrays,
    int device, DeviceRing *(*allocate)(int), std::uint64_t *head) {
  pthread_once(&once, InitializeCondition);
  DeviceRing *ring{nullptr};
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    if (!mailbox->ring) {
      mailbox->ring = allocate(device);
      mailbox->ringDevice = device;
      mailbox->ringOfArrays = arrays;
      mailbox->consumed = 0;
      __atomic_add_fetch(&attachedRings, 1, __ATOMIC_SEQ_CST);
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
    Unlock(mailbox);
  }
  return ring;
}

// Shows the device the messages that did not fit in its ring.
extern "C" void merk_mailbox_drain(MERK_HANDLE_PARAMETERS) {
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    if (mailbox->ring) {
      FillRing(mailbox);
    }
    Unlock(mailbox);
  }
}

// Drops from the list what the device has taken, so that the mailbox's
// oldest message is the next it will take.
extern "C" void merk_mailbox_trim(MERK_HANDLE_PARAMETERS) {
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    if (mailbox->ring) {
      TrimRing(mailbox);
    }
    Unlock(mailbox);
  }
}

// When no region is running: every mailbox goes back to its list alone,
// which still has what the devices did not take. The caller frees the rings.
extern "C" void merk_mailbox_detach_rings(void) {
  if (__atomic_load_n(&attachedRings, __ATOMIC_SEQ_CST) == 0) {
    return;
  }
  std::int64_t count{__atomic_load_n(&slotCount, __ATOMIC_ACQUIRE)};
  for (std::int64_t i{0}; i < count; ++i) {
    Mailbox *mailbox{SlotAt(i)};
    pthread_mutex_lock(&mailbox->lock);
    if (mailbox->open && mailbox->ring) {
      TrimRing(mailbox);
      mailbox->ring = nullptr;
    }
    pthread_mutex_unlock(&mailbox->lock);
  }
  __atomic_store_n(&attachedRings, 0, __ATOMIC_SEQ_CST);
}

// Monitors (design document, "Exceptions, links and monitors"). A process
// that ends must not run its watcher's function, which is in the watcher's
// heap and may be anything: it queues what happened, and the receiver applies
// the function when it reads the message. A monitor is in a hash table by id
// and in its target's list. Locks: a shard's, then the target's waitLock; and
// a mailbox's, then the target's waitLock. A target that is ending takes each
// monitor from its list, then looks for it by id, which a demonitor that holds
// the shard's lock and has found the monitor makes it wait for, so that the
// target's record stays valid for the demonitor.
namespace merkurius::runtime {
struct Monitor {
  std::uint64_t id;
  Process *target;
  Handle watcher;
  Word function;
  Monitor *previous;
  Monitor *next;
  // Whether it is on the target's list.
  bool linked;
  Monitor *nextInShard;
};
} // namespace merkurius::runtime

namespace {
constexpr std::int64_t kDownNormal{1};
constexpr std::int64_t kDownNoProcess{2};
constexpr std::int64_t kDownFailed{3};

struct MonitorShard {
  pthread_mutex_t lock;
  Monitor *first;
};
constexpr std::uint64_t kMonitorShards{256};
MonitorShard monitorShards[kMonitorShards];
std::atomic<std::uint64_t> nextMonitor{1};
pthread_once_t monitorsOnce = PTHREAD_ONCE_INIT;

void InitializeMonitors() {
  for (MonitorShard &shard : monitorShards) {
    pthread_mutex_init(&shard.lock, nullptr);
  }
}

MonitorShard &ShardOf(std::uint64_t id) {
  return monitorShards[id & (kMonitorShards - 1)];
}

// With the shard's lock held: takes the monitor from the table, if it is there.
Monitor *Unhash(MonitorShard &shard, std::uint64_t id) {
  for (Monitor **link{&shard.first}; *link; link = &(*link)->nextInShard) {
    if ((*link)->id == id) {
      Monitor *found{*link};
      *link = found->nextInShard;
      return found;
    }
  }
  return nullptr;
}

bool IsHere(const Handle &handle) {
  pthread_once(&nodeOnce, ChooseNode);
  return handle.nodeHigh == nodeHigh && handle.nodeLow == nodeLow &&
      handle.placeAndSlot >> kPlaceShift == 0;
}

// Queues what happened, unless the watcher's mailbox has closed. No lock held.
void DeliverDown(const Handle &watcher, Word function, std::uint64_t monitor,
    std::int64_t kind, const char *text) {
  Mailbox *mailbox{Find(watcher)};
  if (!mailbox) {
    return;
  }
  Fragment fragment;
  auto *message{static_cast<Message *>(fragment.Allocate(sizeof(Message)))};
  auto *down{static_cast<PendingDown *>(fragment.Allocate(sizeof(PendingDown)))};
  std::size_t bytes{std::strlen(text) + 1};
  auto *copy{static_cast<char *>(fragment.Allocate(bytes))};
  std::memcpy(copy, text, bytes);
  *down = {static_cast<std::int64_t>(monitor), kind, copy};
  message->next = nullptr;
  message->value = function;
  message->down = down;
  Queue(mailbox, message);
  Unlock(mailbox);
}
} // namespace

namespace merkurius::runtime {
void FireMonitorsOf(Process *process, const char *reason) {
  pthread_once(&monitorsOnce, InitializeMonitors);
  for (;;) {
    pthread_mutex_lock(&process->waitLock);
    Monitor *monitor{process->firstMonitor};
    if (!monitor) {
      pthread_mutex_unlock(&process->waitLock);
      return;
    }
    process->firstMonitor = monitor->next;
    if (monitor->next) {
      monitor->next->previous = nullptr;
    }
    monitor->linked = false;
    std::uint64_t id{monitor->id};
    pthread_mutex_unlock(&process->waitLock);
    // A demonitor may have it by now, and free it.
    MonitorShard &shard{ShardOf(id)};
    pthread_mutex_lock(&shard.lock);
    Monitor *mine{Unhash(shard, id)};
    pthread_mutex_unlock(&shard.lock);
    if (mine) {
      DeliverDown(mine->watcher, mine->function, id,
          reason ? kDownFailed : kDownNormal, reason ? reason : "");
      std::free(mine);
    }
  }
}
} // namespace merkurius::runtime

// Starts watching the process that owns the target mailbox. If it has ended,
// which a closed mailbox says, the message is queued at once. Temporary
// (chunk 2, docs/INPROGRESS.md, Before the collector): the
// function is shared with the receiver, as a spawned closure is, if the
// watcher's mailbox is not the caller's.
extern "C" std::int64_t merk_monitor_new(std::int64_t targetHigh,
    std::int64_t targetLow, std::int64_t targetPlaceAndSlot,
    std::int64_t targetGeneration, std::int64_t targetSecret,
    MERK_HANDLE_PARAMETERS, Word function) {
  pthread_once(&once, InitializeCondition);
  pthread_once(&monitorsOnce, InitializeMonitors);
  Handle target{static_cast<Word>(targetHigh), static_cast<Word>(targetLow),
      static_cast<Word>(targetPlaceAndSlot), targetGeneration,
      static_cast<Word>(targetSecret)};
  Handle watcher{MERK_HANDLE};
  if (!IsHere(target) || !IsHere(watcher)) {
    Fail("a monitor across nodes or places is not supported yet", "");
  }
  std::uint64_t id{nextMonitor.fetch_add(1, std::memory_order_relaxed)};
  auto *monitor{static_cast<Monitor *>(Malloc(sizeof(Monitor)))};
  *monitor = {id, nullptr, watcher, function, nullptr, nullptr, false, nullptr};
  MonitorShard &shard{ShardOf(id)};
  pthread_mutex_lock(&shard.lock);
  monitor->nextInShard = shard.first;
  shard.first = monitor;
  pthread_mutex_unlock(&shard.lock);
  if (Mailbox *mailbox{Find(target)}) {
    Process *owner{mailbox->owner};
    if (!owner) {
      Fail("a mailbox of no process cannot be monitored", "");
    }
    // The mailbox's lock keeps it the owner's, so the owner has not ended.
    pthread_mutex_lock(&owner->waitLock);
    monitor->target = owner;
    monitor->next = owner->firstMonitor;
    if (owner->firstMonitor) {
      owner->firstMonitor->previous = monitor;
    }
    owner->firstMonitor = monitor;
    monitor->linked = true;
    pthread_mutex_unlock(&owner->waitLock);
    Unlock(mailbox);
    return static_cast<std::int64_t>(id);
  }
  pthread_mutex_lock(&shard.lock);
  Unhash(shard, id);
  pthread_mutex_unlock(&shard.lock);
  std::free(monitor);
  DeliverDown(watcher, function, id, kDownNoProcess, "");
  return static_cast<std::int64_t>(id);
}

// Stops watching. A message already queued stays, as with Erlang's demonitor
// without its flush option.
extern "C" void merk_monitor_remove(std::int64_t id) {
  pthread_once(&monitorsOnce, InitializeMonitors);
  MonitorShard &shard{ShardOf(static_cast<std::uint64_t>(id))};
  pthread_mutex_lock(&shard.lock);
  if (Monitor *monitor{Unhash(shard, static_cast<std::uint64_t>(id))}) {
    Process *target{monitor->target};
    pthread_mutex_lock(&target->waitLock);
    if (monitor->linked) {
      if (monitor->previous) {
        monitor->previous->next = monitor->next;
      } else {
        target->firstMonitor = monitor->next;
      }
      if (monitor->next) {
        monitor->next->previous = monitor->previous;
      }
    }
    pthread_mutex_unlock(&target->waitLock);
    std::free(monitor);
  }
  pthread_mutex_unlock(&shard.lock);
}

// The message at `position`, which merk_mailbox_wait has found there: 0 and
// its value, or, for what a monitor left, its kind, the function that makes
// the message, the monitor and the text.
extern "C" std::int64_t merk_mailbox_read(MERK_HANDLE_PARAMETERS,
    std::int64_t position, Word *value, Word *function, std::int64_t *monitor,
    const char **text) {
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox || mailbox->count <= position) {
    Fail("a message was read that is not in its mailbox", "");
  }
  Message *message{At(mailbox, position)};
  std::int64_t kind{0};
  *value = 0;
  *function = 0;
  *monitor = 0;
  *text = "";
  if (message->down) {
    kind = message->down->kind;
    *function = message->value;
    *monitor = message->down->monitor;
    *text = message->down->text;
  } else {
    *value = message->value;
  }
  Unlock(mailbox);
  return kind;
}

// The message that the monitor's function made takes the place of what the
// monitor left.
extern "C" void merk_mailbox_resolve(
    MERK_HANDLE_PARAMETERS, std::int64_t position, Word value) {
  if (Mailbox *mailbox{Find(MERK_HANDLE)}) {
    if (mailbox->count > position) {
      Message *message{At(mailbox, position)};
      message->value = value;
      message->down = nullptr;
    }
    Unlock(mailbox);
  }
}

// Registers a mailbox of this node's host under a name, as Erlang's
// register/2 does, which fails if the name is taken or the mailbox has one
// already: 0 if it is registered, 1 if the name is taken, 2 if the mailbox
// has a name, 3 if it is closed or another node's or place's.
extern "C" std::int64_t merk_mailbox_register(
    MERK_HANDLE_PARAMETERS, const char *name) {
  pthread_once(&once, InitializeCondition);
  Mailbox *mailbox{Find(MERK_HANDLE)};
  if (!mailbox) {
    return 3;
  }
  pthread_mutex_lock(&registryLock);
  std::int64_t result{mailbox->name ? 2 : Named(name) ? 1 : 0};
  if (result == 0) {
    std::size_t bytes{std::strlen(name) + 1};
    char *copy{static_cast<char *>(Malloc(bytes))};
    std::memcpy(copy, name, bytes);
    mailbox->name = copy;
  }
  pthread_mutex_unlock(&registryLock);
  Unlock(mailbox);
  return result;
}

// Removes a name, if it is registered.
extern "C" void merk_mailbox_unregister(const char *name) {
  pthread_mutex_lock(&registryLock);
  if (Mailbox *mailbox{Named(name)}) {
    RemoveName(mailbox);
  }
  pthread_mutex_unlock(&registryLock);
}

extern "C" std::int32_t merk_compare_type_infos(Word x, Word y);

// The mailbox registered under a name on this node: 0 and its slot,
// generation and secret if its message type is the type_info's, 1 if no
// mailbox has the name, 2 if its message type is another.
extern "C" std::int64_t merk_mailbox_lookup(const char *name, Word typeInfo,
    std::int64_t *placeAndSlot, std::int64_t *generation,
    std::int64_t *secret) {
  pthread_mutex_lock(&registryLock);
  std::int64_t result{1};
  if (Mailbox *mailbox{Named(name)}) {
    result =
        merk_compare_type_infos(mailbox->messageType, typeInfo) == 0 ? 0 : 2;
    if (result == 0) {
      *placeAndSlot = mailbox->index;
      *generation = mailbox->generation;
      *secret = static_cast<std::int64_t>(mailbox->secret);
    }
  }
  pthread_mutex_unlock(&registryLock);
  return result;
}
