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
// a closed mailbox stays dead when the slot is reused. A send copies the
// message by its type_info into a fragment of its own, outside the heap, and
// queues it; the receiver reads it there, so the fragment joins its heap.
//
// Temporary (chunk 2, docs/INPROGRESS.md): one lock and one condition
// variable for every mailbox, and fragments never freed, while a program is
// one process; chunk 2 gives each process its mailboxes, a lock-free queue
// and a wake-up of its own, and chunk 3's collector merges fragments into the
// receiver's heap.
//
//===----------------------------------------------------------------------===//

#include "walk.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>

namespace {

using namespace merkurius::runtime;

[[noreturn]] void Fail(const char *message, const char *detail) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%s\n", message, detail);
  std::exit(1);
}

void *Malloc(std::size_t bytes) {
  void *block{std::malloc(bytes)};
  if (!block) {
    Fail("out of memory", "");
  }
  return block;
}

// A message's cells, bump-allocated in chunks from malloc, which align them
// to a word at least, as the primary tags need.
class Fragment {
public:
  void *Allocate(std::size_t bytes) {
    bytes = (bytes + 7) & ~std::size_t{7};
    if (static_cast<std::size_t>(limit_ - next_) < bytes) {
      std::size_t size{bytes > kChunkBytes / 4 ? bytes : kChunkBytes};
      next_ = static_cast<char *>(Malloc(size));
      limit_ = next_ + size;
    }
    void *cell{next_};
    next_ += bytes;
    return cell;
  }

private:
  static constexpr std::size_t kChunkBytes{4096};
  char *next_{nullptr};
  char *limit_{nullptr};
};

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
};

struct Mailbox {
  std::int64_t generation;
  bool open;
  Message *head;
  Message **tail;
  std::int64_t count;
  // The next closed slot, to reuse, or -1.
  std::int64_t nextFree;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t arrived;
pthread_once_t once = PTHREAD_ONCE_INIT;
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

// The mailbox of a handle, or null for a closed one. Called with the lock
// held.
Mailbox *Find(std::int64_t slot, std::int64_t generation) {
  if (slot < 0 || slot >= slotCount) {
    return nullptr;
  }
  Mailbox *mailbox{&slots[slot]};
  return mailbox->open && mailbox->generation == generation ? mailbox : nullptr;
}

Message *At(Mailbox *mailbox, std::int64_t position) {
  Message *message{mailbox->head};
  for (; position > 0; --position) {
    message = message->next;
  }
  return message;
}

} // namespace

extern "C" void merk_mailbox_new(std::int64_t *slot, std::int64_t *generation) {
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
  mailbox.head = nullptr;
  mailbox.tail = &mailbox.head;
  mailbox.count = 0;
  mailbox.nextFree = -1;
  *slot = index;
  *generation = mailbox.generation;
  pthread_mutex_unlock(&lock);
}

// Its queued messages are dropped, and its slot gets a new generation, so
// that its handle stays dead when the slot is reused. A receive waiting on
// it wakes and finds nothing.
extern "C" void merk_mailbox_close(std::int64_t slot, std::int64_t generation) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(slot, generation)}) {
    mailbox->open = false;
    ++mailbox->generation;
    mailbox->head = nullptr;
    mailbox->tail = &mailbox->head;
    mailbox->count = 0;
    mailbox->nextFree = firstFree;
    firstFree = slot;
    pthread_cond_broadcast(&arrived);
  }
  pthread_mutex_unlock(&lock);
}

// The copy is made before the lock is taken, since it reads only the
// sender's cells; a message to a closed mailbox is then dropped.
extern "C" void merk_mailbox_send(std::int64_t slot, std::int64_t generation,
    Word typeInfo, Word value) {
  pthread_once(&once, InitializeCondition);
  Fragment fragment;
  auto *message{static_cast<Message *>(fragment.Allocate(sizeof(Message)))};
  message->next = nullptr;
  message->value = CopyValue(value, typeInfo, nullptr, fragment);
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(slot, generation)}) {
    *mailbox->tail = message;
    mailbox->tail = &message->next;
    ++mailbox->count;
    pthread_cond_broadcast(&arrived);
  }
  pthread_mutex_unlock(&lock);
}

// When a receive of this many milliseconds stops waiting, on the monotonic
// clock in nanoseconds, or -1 for a negative number, which waits forever.
extern "C" std::int64_t merk_mailbox_deadline(std::int64_t milliseconds) {
  return milliseconds < 0 ? -1 : Now() + milliseconds * 1000000;
}

// 1 once the mailbox has a message at `position`, 0 if the deadline passes
// first or the mailbox is closed.
extern "C" std::int64_t merk_mailbox_wait(std::int64_t slot,
    std::int64_t generation, std::int64_t position, std::int64_t deadline) {
  pthread_once(&once, InitializeCondition);
  pthread_mutex_lock(&lock);
  std::int64_t result{0};
  for (;;) {
    Mailbox *mailbox{Find(slot, generation)};
    if (!mailbox) {
      break;
    }
    if (mailbox->count > position) {
      result = 1;
      break;
    }
    if (deadline < 0) {
      pthread_cond_wait(&arrived, &lock);
      continue;
    }
    if (Now() >= deadline) {
      break;
    }
    timespec until{static_cast<time_t>(deadline / 1000000000),
        static_cast<long>(deadline % 1000000000)};
    pthread_cond_timedwait(&arrived, &lock, &until);
  }
  pthread_mutex_unlock(&lock);
  return result;
}

// The message at `position`, which merk_mailbox_wait has found there.
extern "C" Word merk_mailbox_message(
    std::int64_t slot, std::int64_t generation, std::int64_t position) {
  pthread_mutex_lock(&lock);
  Mailbox *mailbox{Find(slot, generation)};
  if (!mailbox || mailbox->count <= position) {
    Fail("a message was read that is not in its mailbox", "");
  }
  Word value{At(mailbox, position)->value};
  pthread_mutex_unlock(&lock);
  return value;
}

extern "C" void merk_mailbox_remove(
    std::int64_t slot, std::int64_t generation, std::int64_t position) {
  pthread_mutex_lock(&lock);
  if (Mailbox *mailbox{Find(slot, generation)};
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
