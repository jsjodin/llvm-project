//===-- runtime/node.cpp - Connections to other nodes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TCP connections between nodes, after Erlang's distribution
// (docs/INPROGRESS.md, Remote handles and TCP): listen and connect by
// address, a handshake that exchanges node ids and proves that both nodes
// know the cookie, frames of a 4-byte length
// and a kind, lookups of registered names by their type's fingerprint,
// messages to remote mailboxes, and ticks that find a peer that has gone
// silent.
//
// Each connection has a reader thread, which routes each message it reads
// as a local send would queue it. A sender encodes in its own thread and
// writes the frame under the connection's write lock, so a full socket
// holds it back. A write that makes no progress for a whole tick time, as a
// read that hears nothing for one, loses the connection.
//
// Between two nodes, one connection is current, and messages and lookups go
// on it. The node with the greater id decides which: the newest connection
// that has finished its handshake, whoever started it, so that two nodes
// connecting to each other at once end up with one connection, and a node
// that lost its connection can make a new one before the other has noticed.
// Erlang decides during the handshake, by the names the nodes connect by;
// these ids are learned only in the handshake, so the deciding node tells the
// other with frames:
//
// - It sends `up` on the connection, with the decision's number, and makes
//   it current; it decides on no other connection to that node until this
//   one is acknowledged, so the other node sees the decisions in order.
// - The other node answers `ack` on it, before writing anything else there,
//   makes it current, and stops writing on its older connections to the
//   deciding node, which then see the end of their input.
// - On `ack`, the deciding node stops writing on its older connections too.
//
// Older means decided earlier: a connection that has had no `up` yet is
// left alone, as the deciding node decides on every connection in turn, so
// it is about to become current. Stopping it would leave its `up` with no
// `ack`, and the two nodes with different current connections.
//
// The current connection's reader delivers a message only once every other
// connection to the peer has been read to the end, so messages sent on an
// older connection come first, and the order between two nodes is kept
// across a change of connection.
//
// MERK_TICK_TIME is the tick time in seconds, 60 by default as Erlang's
// net_ticktime.
//
// The cookie is MERK_COOKIE, or else the first line of ~/.merkurius.cookie,
// which is made with 20 random capital letters if it is missing, and must
// be readable by its owner only, as ~/.erlang.cookie. A connection is kept
// only if each node answers the other's random challenge with the HMAC of
// it under the cookie, which is never sent. An answer covers both ids in
// the order of who answers, so that a node's answer on one connection is no
// answer on another, as when it is relayed back to it.
//
// Connections' records are never freed, so that a thread that has found
// one never sees it go; one is a few words per connection ever made.
//
//===----------------------------------------------------------------------===//

#include "node.h"
#include "fragment.h"
#include "heap.h"
#include "sha256.h"
#include <cerrno>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

using namespace merkurius::runtime;

namespace {

enum Kind : std::uint32_t {
  kUp = 1,
  kAck = 2,
  kMessage = 3,
  kLookup = 4,
  kFound = 5,
  kTick = 6,
};

// 0 had no cookie; 1 has the challenges.
constexpr std::uint32_t kVersion{1};
constexpr std::size_t kChallengeBytes{32};
constexpr std::size_t kHelloBytes{24 + kChallengeBytes};
// What an answer's HMAC covers first, so that it is for nothing else.
constexpr char kAnswerLabel[]{"merkurius node answer"};
constexpr std::size_t kCookieBytes{1024};
constexpr std::size_t kHeaderBytes{8};
constexpr std::size_t kHandleBytes{40};
constexpr std::size_t kFoundBytes{16 + kHandleBytes};
constexpr std::size_t kLookupHeadBytes{8 + kFingerprintBytes + 8};
// A reader thread decodes values nested up to kMaxDecodeDepth deep.
constexpr std::size_t kReaderStackBytes{std::size_t{64} << 20};

[[noreturn]] void Fail(const char *message, const char *detail) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%s\n", message, detail);
  std::exit(1);
}

void Put32(std::uint8_t *at, std::uint32_t value) {
  for (int i{0}; i < 4; ++i) {
    at[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

void Put64(std::uint8_t *at, std::uint64_t value) {
  for (int i{0}; i < 8; ++i) {
    at[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

std::uint32_t Get32(const std::uint8_t *at) {
  std::uint32_t value{0};
  for (int i{3}; i >= 0; --i) {
    value = (value << 8) | at[i];
  }
  return value;
}

std::uint64_t Get64(const std::uint8_t *at) {
  std::uint64_t value{0};
  for (int i{7}; i >= 0; --i) {
    value = (value << 8) | at[i];
  }
  return value;
}

void PutHandle(std::uint8_t *at, const Handle &handle) {
  Put64(at, handle.nodeHigh);
  Put64(at + 8, handle.nodeLow);
  Put64(at + 16, handle.placeAndSlot);
  Put64(at + 24, static_cast<std::uint64_t>(handle.generation));
  Put64(at + 32, handle.secret);
}

Handle GetHandle(const std::uint8_t *at) {
  return {Get64(at), Get64(at + 8), Get64(at + 16),
      static_cast<std::int64_t>(Get64(at + 24)), Get64(at + 32)};
}

struct Connection {
  // Set before the record is published, and never changed.
  Word peerHigh;
  Word peerLow;
  Connection *next;
  // Under `lock`. Its reader has read it to the end and closed it: `done`
  // is also read without the lock, by the tick thread.
  bool current;
  bool stopped;
  bool done;
  // The deciding node's number for it, once it has sent `up`; on the
  // deciding node, whether the other has acknowledged it.
  std::uint64_t decision;
  bool upSent;
  bool acknowledged;
  // Held while a frame is written. `writable` is under it, and the socket
  // is closed only with it held, so a writer that finds it writable can use
  // the socket.
  pthread_mutex_t writeLock;
  bool writable;
  // Guards closing the socket against the tick thread's shutdown, which
  // must not wait for a writer blocked on a full socket.
  pthread_mutex_t socketLock;
  int socket;
  // Nanoseconds on the monotonic clock, read and written atomically.
  std::int64_t lastHeard;
  std::int64_t lastSent;
};

// A lookup waiting for its answer.
struct Pending {
  Pending *next;
  std::uint64_t request;
  Word peerHigh;
  Word peerLow;
  bool answered;
  std::int64_t status;
  Handle handle;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// Broadcast when a connection becomes current or done, or a lookup is
// answered.
pthread_cond_t changed;
pthread_once_t once = PTHREAD_ONCE_INIT;
// Every connection ever made, newest first; the head is read atomically.
Connection *connections{nullptr};
std::uint64_t decisions{0};
std::uint64_t requests{0};
Pending *pendings{nullptr};
std::int64_t tickTime{60 * std::int64_t{1000000000}};

std::int64_t Now() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::int64_t>(now.tv_sec) * 1000000000 + now.tv_nsec;
}

// Waits on `changed` until the deadline, or for ever if it is negative.
// Called with the lock held.
void WaitUntil(std::int64_t deadline) {
  if (deadline < 0) {
    pthread_cond_wait(&changed, &lock);
    return;
  }
  timespec until{static_cast<time_t>(deadline / 1000000000),
      static_cast<long>(deadline % 1000000000)};
  pthread_cond_timedwait(&changed, &lock, &until);
}

void *Ticker(void *);

void StartThread(void *(*body)(void *), void *argument) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  pthread_attr_setstacksize(&attributes, kReaderStackBytes);
  pthread_t thread;
  if (pthread_create(&thread, &attributes, body, argument) != 0) {
    Fail("cannot start a thread for a connection", "");
  }
  pthread_attr_destroy(&attributes);
}

void Initialize() {
  pthread_condattr_t attributes;
  pthread_condattr_init(&attributes);
  pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC);
  pthread_cond_init(&changed, &attributes);
  pthread_condattr_destroy(&attributes);
  if (const char *text{std::getenv("MERK_TICK_TIME")}) {
    char *end;
    long seconds{std::strtol(text, &end, 10)};
    if (*text == '\0' || *end != '\0' || seconds <= 0 || seconds > 86400) {
      Fail("MERK_TICK_TIME is not a number of seconds from 1 to 86400: ",
          text);
    }
    tickTime = seconds * std::int64_t{1000000000};
  }
  StartThread(Ticker, nullptr);
}

// The cookie, or why there is none.
pthread_once_t cookieOnce = PTHREAD_ONCE_INIT;
char cookie[kCookieBytes];
std::size_t cookieBytes{0};
char cookieError[4096 + 128];

void CookieProblem(const char *where, const char *problem) {
  std::snprintf(cookieError, sizeof cookieError, "%s: %s", where, problem);
}

// The file's first line, without the spaces after it, into `cookie`; it is
// made first if it is missing. False, with the problem in cookieError, if
// it cannot be read, is not its owner's alone, or is empty.
bool ReadCookieFile(const char *path) {
  int file{open(path, O_RDONLY | O_CLOEXEC)};
  if (file < 0 && errno == ENOENT) {
    int made{open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0400)};
    if (made >= 0) {
      char letters[20];
      std::uint8_t random[sizeof letters];
      while (getrandom(random, sizeof random, 0) != sizeof random) {
      }
      for (std::size_t i{0}; i < sizeof letters; ++i) {
        // 256 is not a multiple of 26, but the bias does not matter for a
        // cookie of 20 letters.
        letters[i] = static_cast<char>('A' + random[i] % 26);
      }
      bool written{write(made, letters, sizeof letters) ==
          static_cast<ssize_t>(sizeof letters)};
      close(made);
      if (!written) {
        CookieProblem(path, "cannot write the new cookie");
        return false;
      }
    } else if (errno != EEXIST) {
      CookieProblem(path, std::strerror(errno));
      return false;
    }
    // Made by this node or, at the same time, by another.
    file = open(path, O_RDONLY | O_CLOEXEC);
  }
  if (file < 0) {
    CookieProblem(path, std::strerror(errno));
    return false;
  }
  struct stat status;
  if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_uid != geteuid() || (status.st_mode & 077) != 0) {
    close(file);
    CookieProblem(path, "the cookie file must be its owner's and readable by "
                        "its owner only");
    return false;
  }
  ssize_t got{read(file, cookie, sizeof cookie)};
  close(file);
  if (got < 0) {
    CookieProblem(path, std::strerror(errno));
    return false;
  }
  std::size_t bytes{0};
  while (bytes < static_cast<std::size_t>(got) && cookie[bytes] != '\n') {
    ++bytes;
  }
  while (bytes > 0 && (cookie[bytes - 1] == ' ' || cookie[bytes - 1] == '\t' ||
                          cookie[bytes - 1] == '\r')) {
    --bytes;
  }
  if (bytes == 0) {
    CookieProblem(path, "the cookie file is empty");
    return false;
  }
  cookieBytes = bytes;
  return true;
}

void LoadCookie() {
  if (const char *text{std::getenv("MERK_COOKIE")}) {
    std::size_t bytes{std::strlen(text)};
    if (bytes == 0 || bytes > sizeof cookie) {
      CookieProblem("MERK_COOKIE", bytes == 0 ? "the cookie is empty"
                                              : "the cookie is too long");
      return;
    }
    std::memcpy(cookie, text, bytes);
    cookieBytes = bytes;
    return;
  }
  const char *home{std::getenv("HOME")};
  if (!home || !*home) {
    CookieProblem("~/.merkurius.cookie",
        "HOME is not set, and neither is MERK_COOKIE");
    return;
  }
  char path[4096];
  if (std::snprintf(path, sizeof path, "%s/.merkurius.cookie", home) >=
      static_cast<int>(sizeof path)) {
    CookieProblem("~/.merkurius.cookie", "HOME is too long");
    return;
  }
  ReadCookieFile(path);
}

// Null once the cookie is known, or why it is not.
const char *CookieError() {
  pthread_once(&cookieOnce, LoadCookie);
  return cookieBytes > 0 ? nullptr : cookieError;
}

// The HMAC under the cookie of the label, a challenge, and the answering
// node's id and then the other's.
void Answer(const std::uint8_t *challenge, Word answerHigh, Word answerLow,
    Word otherHigh, Word otherLow, std::uint8_t answer[Sha256::kDigestBytes]) {
  std::uint8_t data[sizeof kAnswerLabel + kChallengeBytes + 32];
  std::memcpy(data, kAnswerLabel, sizeof kAnswerLabel);
  std::uint8_t *at{data + sizeof kAnswerLabel};
  std::memcpy(at, challenge, kChallengeBytes);
  at += kChallengeBytes;
  Put64(at, answerHigh);
  Put64(at + 8, answerLow);
  Put64(at + 16, otherHigh);
  Put64(at + 24, otherLow);
  HmacSha256(cookie, cookieBytes, data, sizeof data, answer);
}

// Whether two answers are equal, in a time that does not depend on where
// they differ.
bool SameAnswer(const std::uint8_t *a, const std::uint8_t *b) {
  std::uint8_t differ{0};
  for (std::size_t i{0}; i < Sha256::kDigestBytes; ++i) {
    differ |= a[i] ^ b[i];
  }
  return differ == 0;
}

bool IsPeer(const Connection *connection, Word high, Word low) {
  return connection->peerHigh == high && connection->peerLow == low;
}

// Whether this node decides between its connections to the peer.
bool Decides(Word peerHigh, Word peerLow) {
  Word high, low;
  ThisNode(high, low);
  return high != peerHigh ? high > peerHigh : low > peerLow;
}

Connection *First() { return __atomic_load_n(&connections, __ATOMIC_ACQUIRE); }

// Called with the lock held.
Connection *CurrentOf(Word high, Word low) {
  for (Connection *c{First()}; c; c = c->next) {
    if (c->current && IsPeer(c, high, low)) {
      return c;
    }
  }
  return nullptr;
}

// A connection to the peer, other than `except`, that is not done, and that
// is still to be decided on if `undecided`. Called with the lock held.
Connection *OtherOf(
    const Connection *except, Word high, Word low, bool undecided) {
  for (Connection *c{First()}; c; c = c->next) {
    if (c != except && !c->done && IsPeer(c, high, low) &&
        (!undecided || (!c->current && !c->stopped))) {
      return c;
    }
  }
  return nullptr;
}

// Sends every byte of the parts; false on an error, or a send that makes no
// progress for a tick time (SO_SNDTIMEO).
bool SendAll(int socket, iovec *parts, int count) {
  msghdr message{};
  message.msg_iov = parts;
  message.msg_iovlen = count;
  for (;;) {
    while (message.msg_iovlen > 0 && message.msg_iov[0].iov_len == 0) {
      ++message.msg_iov;
      --message.msg_iovlen;
    }
    if (message.msg_iovlen == 0) {
      return true;
    }
    ssize_t sent{sendmsg(socket, &message, MSG_NOSIGNAL)};
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    auto left{static_cast<std::size_t>(sent)};
    while (left > 0) {
      iovec &part{message.msg_iov[0]};
      std::size_t taken{left < part.iov_len ? left : part.iov_len};
      part.iov_base = static_cast<char *>(part.iov_base) + taken;
      part.iov_len -= taken;
      left -= taken;
      if (part.iov_len == 0) {
        ++message.msg_iov;
        --message.msg_iovlen;
      }
    }
  }
}

// As recv, looking for what is there for a while before it blocks.
ssize_t Receive(int socket, void *into, std::size_t bytes) {
  if (std::int64_t spin{SpinNanoseconds()}; spin != 0) {
    std::int64_t end{Now() + spin};
    do {
      ssize_t got{recv(socket, into, bytes, MSG_DONTWAIT)};
      if (got >= 0 || errno != EAGAIN) {
        return got;
      }
      sched_yield();
    } while (Now() < end);
  }
  return recv(socket, into, bytes, 0);
}

bool ReadAll(int socket, void *into, std::size_t bytes) {
  auto *at{static_cast<char *>(into)};
  while (bytes > 0) {
    ssize_t got{Receive(socket, at, bytes)};
    if (got < 0 && errno == EINTR) {
      continue;
    }
    if (got <= 0) {
      return false;
    }
    at += got;
    bytes -= static_cast<std::size_t>(got);
  }
  return true;
}

// Writes a frame: false if the connection takes no more, or the write
// fails, which loses the connection. Called with the write lock held.
bool WriteLocked(Connection *c, std::uint32_t kind, const void *head,
    std::size_t headBytes, const void *body, std::size_t bodyBytes) {
  if (!c->writable) {
    return false;
  }
  std::uint8_t header[kHeaderBytes];
  Put32(header, static_cast<std::uint32_t>(headBytes + bodyBytes));
  Put32(header + 4, kind);
  iovec parts[3]{{header, kHeaderBytes}, {const_cast<void *>(head), headBytes},
      {const_cast<void *>(body), bodyBytes}};
  if (!SendAll(c->socket, parts, 3)) {
    c->writable = false;
    shutdown(c->socket, SHUT_RDWR);
    return false;
  }
  __atomic_store_n(&c->lastSent, Now(), __ATOMIC_RELAXED);
  return true;
}

bool Write(Connection *c, std::uint32_t kind, const void *head,
    std::size_t headBytes, const void *body = nullptr,
    std::size_t bodyBytes = 0) {
  pthread_mutex_lock(&c->writeLock);
  bool written{WriteLocked(c, kind, head, headBytes, body, bodyBytes)};
  pthread_mutex_unlock(&c->writeLock);
  return written;
}

// This side writes nothing more on it; the peer sees the end of its input.
void StopWriting(Connection *c) {
  pthread_mutex_lock(&c->writeLock);
  if (c->writable) {
    c->writable = false;
    shutdown(c->socket, SHUT_WR);
  }
  pthread_mutex_unlock(&c->writeLock);
}

// Stops writing on the connections to c's peer that were decided before c.
void StopOthers(Connection *c) {
  for (;;) {
    pthread_mutex_lock(&lock);
    Connection *other{nullptr};
    for (Connection *x{First()}; x; x = x->next) {
      if (x != c && !x->current && !x->stopped && !x->done &&
          x->decision != 0 && x->decision < c->decision &&
          IsPeer(x, c->peerHigh, c->peerLow)) {
        other = x;
        break;
      }
    }
    if (other) {
      other->stopped = true;
      pthread_cond_broadcast(&changed);
    }
    pthread_mutex_unlock(&lock);
    if (!other) {
      return;
    }
    StopWriting(other);
  }
}

// Makes c the peer's current connection. Called with the lock held.
void MakeCurrent(Connection *c) {
  for (Connection *x{First()}; x; x = x->next) {
    if (IsPeer(x, c->peerHigh, c->peerLow)) {
      x->current = false;
    }
  }
  c->current = true;
  pthread_cond_broadcast(&changed);
}

// On the deciding node: c becomes current once no earlier decision waits
// for its acknowledgement.
void Decide(Connection *c) {
  pthread_mutex_lock(&lock);
  for (;;) {
    bool waiting{false};
    for (Connection *x{First()}; x; x = x->next) {
      if (x != c && x->upSent && !x->acknowledged && !x->done &&
          IsPeer(x, c->peerHigh, c->peerLow)) {
        waiting = true;
        break;
      }
    }
    if (!waiting || c->done) {
      break;
    }
    WaitUntil(-1);
  }
  std::uint8_t number[8];
  c->decision = ++decisions;
  Put64(number, c->decision);
  c->upSent = true;
  if (Write(c, kUp, number, sizeof number)) {
    MakeCurrent(c);
  }
  pthread_mutex_unlock(&lock);
}

// The frame goes on the peer's current connection, waiting while a
// connection to it is still to be decided on; false if there is none.
bool SendToPeer(Word high, Word low, std::uint32_t kind, const void *head,
    std::size_t headBytes, const void *body, std::size_t bodyBytes) {
  for (;;) {
    pthread_mutex_lock(&lock);
    Connection *c{CurrentOf(high, low)};
    while (!c && OtherOf(nullptr, high, low, true)) {
      WaitUntil(-1);
      c = CurrentOf(high, low);
    }
    pthread_mutex_unlock(&lock);
    if (!c) {
      return false;
    }
    if (Write(c, kind, head, headBytes, body, bodyBytes)) {
      return true;
    }
    // It was replaced or lost; the next current takes the frame.
    pthread_mutex_lock(&lock);
    if (c->current) {
      c->current = false;
      pthread_cond_broadcast(&changed);
    }
    pthread_mutex_unlock(&lock);
  }
}

void Lose(Connection *c) {
  pthread_mutex_lock(&c->socketLock);
  if (c->socket >= 0) {
    shutdown(c->socket, SHUT_RDWR);
  }
  pthread_mutex_unlock(&c->socketLock);
}

// Sends ticks on idle connections and loses silent ones.
void *Ticker(void *) {
  for (;;) {
    timespec quarter{static_cast<time_t>(tickTime / 4 / 1000000000),
        static_cast<long>(tickTime / 4 % 1000000000)};
    while (clock_nanosleep(CLOCK_MONOTONIC, 0, &quarter, &quarter) == EINTR) {
    }
    std::int64_t now{Now()};
    for (Connection *c{First()}; c; c = c->next) {
      if (__atomic_load_n(&c->done, __ATOMIC_ACQUIRE)) {
        continue;
      }
      if (now - __atomic_load_n(&c->lastHeard, __ATOMIC_RELAXED) >= tickTime) {
        Lose(c);
        continue;
      }
      if (now - __atomic_load_n(&c->lastSent, __ATOMIC_RELAXED) >=
              tickTime / 4 &&
          pthread_mutex_trylock(&c->writeLock) == 0) {
        WriteLocked(c, kTick, nullptr, 0, nullptr, 0);
        pthread_mutex_unlock(&c->writeLock);
      }
    }
  }
  return nullptr;
}

// The end of a connection, from its reader: nothing more is read or
// written, and the socket is closed.
void Finish(Connection *c) {
  shutdown(c->socket, SHUT_RDWR);
  pthread_mutex_lock(&c->writeLock);
  c->writable = false;
  pthread_mutex_lock(&c->socketLock);
  close(c->socket);
  c->socket = -1;
  pthread_mutex_unlock(&c->socketLock);
  pthread_mutex_unlock(&c->writeLock);
  pthread_mutex_lock(&lock);
  c->current = false;
  __atomic_store_n(&c->done, true, __ATOMIC_RELEASE);
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&lock);
}

// `up` on the other node: c becomes current, unless a later decision has.
// False for a frame the deciding node would not send.
bool Up(Connection *c, const std::uint8_t *frame, std::size_t size) {
  if (size != 8 || Decides(c->peerHigh, c->peerLow)) {
    return false;
  }
  std::uint64_t decision{Get64(frame)};
  pthread_mutex_lock(&lock);
  Connection *current{CurrentOf(c->peerHigh, c->peerLow)};
  bool later{!current || current->decision < decision};
  c->decision = decision;
  if (later) {
    // Acknowledged before it is current, so that nothing is written on it
    // before the `ack`.
    if (Write(c, kAck, nullptr, 0)) {
      MakeCurrent(c);
    }
  } else {
    c->stopped = true;
    pthread_cond_broadcast(&changed);
  }
  pthread_mutex_unlock(&lock);
  if (later) {
    StopOthers(c);
  } else {
    StopWriting(c);
  }
  return true;
}

bool Ack(Connection *c, std::size_t size) {
  if (size != 0 || !Decides(c->peerHigh, c->peerLow)) {
    return false;
  }
  pthread_mutex_lock(&lock);
  bool expected{c->upSent && !c->acknowledged};
  c->acknowledged = true;
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&lock);
  if (expected) {
    StopOthers(c);
  }
  return expected;
}

// Waits, on the current connection, until every other connection to the
// peer is read to the end, so that messages sent on them come first.
void WaitForOthers(Connection *c) {
  pthread_mutex_lock(&lock);
  while (c->current && OtherOf(c, c->peerHigh, c->peerLow, false)) {
    WaitUntil(-1);
  }
  pthread_mutex_unlock(&lock);
}

bool Message(Connection *c, const std::uint8_t *frame, std::size_t size) {
  if (size < kHandleBytes) {
    return false;
  }
  WaitForOthers(c);
  return Route(GetHandle(frame), frame + kHandleBytes, size - kHandleBytes) !=
      Delivery::Undecodable;
}

bool Lookup(Connection *c, const std::uint8_t *frame, std::size_t size) {
  if (size < kLookupHeadBytes) {
    return false;
  }
  std::uint64_t length{Get64(frame + 8 + kFingerprintBytes)};
  std::size_t left{size - kLookupHeadBytes};
  if (length > left || (length + 7) / 8 * 8 != left) {
    return false;
  }
  const std::uint8_t *text{frame + kLookupHeadBytes};
  if (std::memchr(text, '\0', length)) {
    return false;
  }
  auto *name{static_cast<char *>(Malloc(length + 1))};
  std::memcpy(name, text, length);
  name[length] = '\0';
  Handle found{};
  std::int64_t status{LookupForPeer(name, frame + 8, found)};
  std::free(name);
  std::uint8_t answer[kFoundBytes]{};
  std::memcpy(answer, frame, 8);
  Put64(answer + 8, static_cast<std::uint64_t>(status));
  if (status == 0) {
    PutHandle(answer + 16, found);
  }
  SendToPeer(c->peerHigh, c->peerLow, kFound, answer, sizeof answer, nullptr, 0);
  return true;
}

bool Found(Connection *c, const std::uint8_t *frame, std::size_t size) {
  if (size != kFoundBytes) {
    return false;
  }
  std::uint64_t request{Get64(frame)};
  std::uint64_t status{Get64(frame + 8)};
  Handle handle{GetHandle(frame + 16)};
  // A handle of another node, or of a place, is not one of its names'.
  if (status > 2 || (status == 0 &&
                        (!IsPeer(c, handle.nodeHigh, handle.nodeLow) ||
                            handle.placeAndSlot >> kPlaceShift != 0))) {
    return false;
  }
  pthread_mutex_lock(&lock);
  for (Pending *p{pendings}; p; p = p->next) {
    if (p->request == request && !p->answered &&
        IsPeer(c, p->peerHigh, p->peerLow)) {
      p->answered = true;
      p->status = static_cast<std::int64_t>(status);
      p->handle = handle;
      pthread_cond_broadcast(&changed);
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  return true;
}

// Reads frames until the connection ends, or a frame is not one the peer
// may send, which ends it too.
void Read(Connection *c) {
  std::uint8_t *buffer{nullptr};
  std::size_t capacity{0};
  for (;;) {
    std::uint8_t header[kHeaderBytes];
    if (!ReadAll(c->socket, header, sizeof header)) {
      break;
    }
    std::size_t size{Get32(header)};
    std::uint32_t kind{Get32(header + 4)};
    // The buffer grows as the bytes arrive, so a length alone allocates
    // nothing.
    std::size_t got{0};
    bool complete{true};
    while (got < size) {
      if (got == capacity) {
        std::size_t grown{capacity < 65536 ? 65536 : 2 * capacity};
        capacity = grown < size ? grown : size;
        buffer = static_cast<std::uint8_t *>(std::realloc(buffer, capacity));
        if (!buffer) {
          Fail("out of memory", "");
        }
      }
      std::size_t chunk{(capacity < size ? capacity : size) - got};
      if (!ReadAll(c->socket, buffer + got, chunk)) {
        complete = false;
        break;
      }
      got += chunk;
    }
    if (!complete) {
      break;
    }
    __atomic_store_n(&c->lastHeard, Now(), __ATOMIC_RELAXED);
    bool valid{false};
    switch (kind) {
    case kUp:
      valid = Up(c, buffer, size);
      break;
    case kAck:
      valid = Ack(c, size);
      break;
    case kMessage:
      valid = Message(c, buffer, size);
      break;
    case kLookup:
      valid = Lookup(c, buffer, size);
      break;
    case kFound:
      valid = Found(c, buffer, size);
      break;
    case kTick:
      valid = size == 0;
      break;
    }
    if (!valid) {
      break;
    }
  }
  std::free(buffer);
  Finish(c);
}

void *Reader(void *argument) {
  Read(static_cast<Connection *>(argument));
  return nullptr;
}

void SetTimeout(int socket, int option, std::int64_t nanoseconds) {
  timeval time{static_cast<time_t>(nanoseconds / 1000000000),
      static_cast<suseconds_t>(nanoseconds % 1000000000 / 1000)};
  setsockopt(socket, SOL_SOCKET, option, &time, sizeof time);
}

// Both sides send `MERK`, the version, their id and a random challenge, and
// read the other's; then each sends its answer to the other's challenge,
// and checks the other's answer to its own: null and the peer's id, or why
// the connection is refused. A handshake that hears nothing for a tick time
// fails.
const char *Handshake(int socket, Word &peerHigh, Word &peerLow) {
  int one{1};
  setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  SetTimeout(socket, SO_SNDTIMEO, tickTime);
  SetTimeout(socket, SO_RCVTIMEO, tickTime);
  Word high, low;
  ThisNode(high, low);
  std::uint8_t hello[kHelloBytes];
  std::memcpy(hello, "MERK", 4);
  Put32(hello + 4, kVersion);
  Put64(hello + 8, high);
  Put64(hello + 16, low);
  std::uint8_t *challenge{hello + 24};
  while (getrandom(challenge, kChallengeBytes, 0) !=
      static_cast<ssize_t>(kChallengeBytes)) {
  }
  iovec part{hello, sizeof hello};
  if (!SendAll(socket, &part, 1)) {
    return "the other node did not take the handshake";
  }
  std::uint8_t theirs[kHelloBytes];
  if (!ReadAll(socket, theirs, sizeof theirs)) {
    return "the other node did not answer the handshake";
  }
  if (std::memcmp(theirs, "MERK", 4) != 0) {
    return "the other side is not a Merkurius node";
  }
  if (Get32(theirs + 4) != kVersion) {
    return "the other node speaks another version of the protocol";
  }
  peerHigh = Get64(theirs + 8);
  peerLow = Get64(theirs + 16);
  if (peerHigh == high && peerLow == low) {
    return "a node cannot connect to itself";
  }
  std::uint8_t answer[Sha256::kDigestBytes];
  Answer(theirs + 24, high, low, peerHigh, peerLow, answer);
  iovec answerPart{answer, sizeof answer};
  if (!SendAll(socket, &answerPart, 1)) {
    return "the other node did not take the handshake";
  }
  std::uint8_t theirAnswer[Sha256::kDigestBytes];
  if (!ReadAll(socket, theirAnswer, sizeof theirAnswer)) {
    return "the other node did not answer the handshake";
  }
  std::uint8_t expected[Sha256::kDigestBytes];
  Answer(challenge, peerHigh, peerLow, high, low, expected);
  if (!SameAnswer(theirAnswer, expected)) {
    return "the other node has another cookie";
  }
  SetTimeout(socket, SO_RCVTIMEO, 0);
  return nullptr;
}

// Publishes a connection whose handshake is done, and on the deciding node
// decides on it.
Connection *Register(int socket, Word peerHigh, Word peerLow) {
  auto *c{static_cast<Connection *>(Malloc(sizeof(Connection)))};
  c->peerHigh = peerHigh;
  c->peerLow = peerLow;
  c->current = false;
  c->stopped = false;
  c->done = false;
  c->decision = 0;
  c->upSent = false;
  c->acknowledged = false;
  pthread_mutex_init(&c->writeLock, nullptr);
  c->writable = true;
  pthread_mutex_init(&c->socketLock, nullptr);
  c->socket = socket;
  c->lastHeard = Now();
  c->lastSent = c->lastHeard;
  pthread_mutex_lock(&lock);
  c->next = connections;
  __atomic_store_n(&connections, c, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&lock);
  if (Decides(peerHigh, peerLow)) {
    Decide(c);
  }
  return c;
}

// An accepted connection's thread: the handshake, then its reader.
void *Join(void *argument) {
  int socket{static_cast<int>(reinterpret_cast<std::intptr_t>(argument))};
  Word peerHigh, peerLow;
  if (Handshake(socket, peerHigh, peerLow)) {
    close(socket);
    return nullptr;
  }
  Read(Register(socket, peerHigh, peerLow));
  return nullptr;
}

void *Accept(void *argument) {
  int listener{static_cast<int>(reinterpret_cast<std::intptr_t>(argument))};
  for (;;) {
    int socket{accept4(listener, nullptr, nullptr, SOCK_CLOEXEC)};
    if (socket < 0) {
      if (errno != EINTR && errno != ECONNABORTED) {
        // Out of descriptors or memory, say: try again a little later.
        timespec pause{0, 100000000};
        nanosleep(&pause, nullptr);
      }
      continue;
    }
    StartThread(Join, reinterpret_cast<void *>(std::intptr_t{socket}));
  }
  return nullptr;
}

// "what: detail" on the calling thread's heap, for Mercury.
const char *HeapText(const char *what, const char *detail) {
  std::size_t a{std::strlen(what)};
  std::size_t b{std::strlen(detail)};
  std::size_t bytes{a + 2 + b};
  auto *text{static_cast<char *>(Allocate((bytes + 8) & ~std::size_t{7}))};
  std::memcpy(text, what, a);
  std::memcpy(text + a, ": ", 2);
  std::memcpy(text + a + 2, detail, b);
  text[bytes] = '\0';
  return text;
}

const char *SystemError(const char *what, int error) {
  char buffer[128];
  return HeapText(what, strerror_r(error, buffer, sizeof buffer));
}

// Splits "host:port" at its last colon, and "[host]:port" too; false if
// there is no colon, or a part is too long.
bool SplitAddress(const char *address, char (&host)[256], char (&port)[32]) {
  const char *colon{std::strrchr(address, ':')};
  if (!colon) {
    return false;
  }
  const char *start{address};
  const char *end{colon};
  if (end - start >= 2 && *start == '[' && end[-1] == ']') {
    ++start;
    --end;
  }
  auto hostBytes{static_cast<std::size_t>(end - start)};
  std::size_t portBytes{std::strlen(colon + 1)};
  if (hostBytes >= sizeof host || portBytes >= sizeof port || portBytes == 0) {
    return false;
  }
  std::memcpy(host, start, hostBytes);
  host[hostBytes] = '\0';
  std::memcpy(port, colon + 1, portBytes + 1);
  return true;
}

// The addresses of "host:port": null, or why there are none.
const char *Addresses(const char *address, bool passive, addrinfo *&found) {
  char host[256];
  char port[32];
  if (!SplitAddress(address, host, port)) {
    return HeapText("not an address of the form host:port", address);
  }
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = passive ? AI_PASSIVE : 0;
  int error{getaddrinfo(host[0] ? host : nullptr, port, &hints, &found)};
  return error ? HeapText(address, gai_strerror(error)) : nullptr;
}

} // namespace

namespace merkurius::runtime {

Delivery Route(
    const Handle &handle, const std::uint8_t *bytes, std::size_t size) {
  if (IsThisNode(handle.nodeHigh, handle.nodeLow)) {
    // A place this node does not have finds no mailbox: Temporary (GPU
    // processes, docs/INPROGRESS.md): until the bridge, that is every place
    // but the host.
    return DeliverEncoded(handle, bytes, size);
  }
  pthread_once(&once, Initialize);
  if (size > 0xFFFFFFFFu - kHandleBytes) {
    Fail("a message is too large to send to another node: its encoding is "
         "over 4 GiB",
        "");
  }
  std::uint8_t head[kHandleBytes];
  PutHandle(head, handle);
  return SendToPeer(handle.nodeHigh, handle.nodeLow, kMessage, head,
             sizeof head, bytes, size)
      ? Delivery::Delivered
      : Delivery::Dropped;
}

} // namespace merkurius::runtime

// Listens on "host:port", where port 0 picks a free one: 0 and the port, or
// 1 and why not.
extern "C" std::int64_t merk_node_listen(
    const char *address, std::int64_t *port, const char **error) {
  pthread_once(&once, Initialize);
  if (const char *problem{CookieError()}) {
    *error = HeapText("no cookie", problem);
    return 1;
  }
  addrinfo *found;
  if ((*error = Addresses(address, true, found))) {
    return 1;
  }
  int listener{-1};
  int last{0};
  for (addrinfo *a{found}; a && listener < 0; a = a->ai_next) {
    listener = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, 0);
    if (listener < 0) {
      last = errno;
      continue;
    }
    int one{1};
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(listener, a->ai_addr, a->ai_addrlen) != 0 ||
        ::listen(listener, 128) != 0) {
      last = errno;
      close(listener);
      listener = -1;
    }
  }
  freeaddrinfo(found);
  if (listener < 0) {
    *error = SystemError(address, last);
    return 1;
  }
  sockaddr_storage bound{};
  socklen_t bytes{sizeof bound};
  getsockname(listener, reinterpret_cast<sockaddr *>(&bound), &bytes);
  *port = bound.ss_family == AF_INET6
      ? ntohs(reinterpret_cast<sockaddr_in6 *>(&bound)->sin6_port)
      : ntohs(reinterpret_cast<sockaddr_in *>(&bound)->sin_port);
  StartThread(Accept, reinterpret_cast<void *>(std::intptr_t{listener}));
  return 0;
}

// Connects to the node listening on "host:port": 0 and its id once the
// connection is current, or 1 and why not.
extern "C" std::int64_t merk_node_connect(const char *address,
    std::int64_t *high, std::int64_t *low, const char **error) {
  pthread_once(&once, Initialize);
  if (const char *problem{CookieError()}) {
    *error = HeapText("no cookie", problem);
    return 1;
  }
  addrinfo *found;
  if ((*error = Addresses(address, false, found))) {
    return 1;
  }
  int connected{-1};
  int last{0};
  for (addrinfo *a{found}; a && connected < 0; a = a->ai_next) {
    connected = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, 0);
    if (connected < 0) {
      last = errno;
      continue;
    }
    if (::connect(connected, a->ai_addr, a->ai_addrlen) != 0) {
      last = errno;
      close(connected);
      connected = -1;
    }
  }
  freeaddrinfo(found);
  if (connected < 0) {
    *error = SystemError(address, last);
    return 1;
  }
  Word peerHigh, peerLow;
  if (const char *refused{Handshake(connected, peerHigh, peerLow)}) {
    close(connected);
    *error = HeapText(address, refused);
    return 1;
  }
  Connection *c{Register(connected, peerHigh, peerLow)};
  StartThread(Reader, c);
  // The other node decides; wait for a current connection to it, this one
  // or one that replaced it.
  std::int64_t deadline{Now() + tickTime};
  pthread_mutex_lock(&lock);
  while (!CurrentOf(peerHigh, peerLow) && !c->done && Now() < deadline) {
    WaitUntil(deadline);
  }
  bool connectedNow{CurrentOf(peerHigh, peerLow) != nullptr};
  pthread_mutex_unlock(&lock);
  if (!connectedNow) {
    *error = HeapText(address, "the connection closed during the handshake");
    return 1;
  }
  *high = static_cast<std::int64_t>(peerHigh);
  *low = static_cast<std::int64_t>(peerLow);
  return 0;
}

// The mailbox registered under a name on another node, whose messages have
// the type: 0 and its handle's slot, generation and secret; 1 if no mailbox
// has the name, 2 if its type is another, 3 if this node is not connected
// to that one, 4 if the connection is lost before the answer, and 5 if the
// answer does not come within the milliseconds, if they are not negative.
extern "C" std::int64_t merk_node_lookup(std::int64_t high, std::int64_t low,
    const char *name, Word typeInfo, std::int64_t milliseconds,
    std::int64_t *placeAndSlot, std::int64_t *generation,
    std::int64_t *secret) {
  pthread_once(&once, Initialize);
  auto peerHigh{static_cast<Word>(high)};
  auto peerLow{static_cast<Word>(low)};
  std::int64_t deadline{milliseconds < 0 ? -1 : Now() + milliseconds * 1000000};
  std::size_t length{std::strlen(name)};
  std::size_t padded{(length + 7) / 8 * 8};
  if (padded > 0xFFFF0000u) {
    return 1;
  }
  auto *frame{static_cast<std::uint8_t *>(Malloc(kLookupHeadBytes + padded))};
  std::memset(frame, 0, kLookupHeadBytes + padded);
  Pending pending{};
  pending.peerHigh = peerHigh;
  pending.peerLow = peerLow;
  pthread_mutex_lock(&lock);
  pending.request = ++requests;
  pending.next = pendings;
  pendings = &pending;
  pthread_mutex_unlock(&lock);
  Put64(frame, pending.request);
  Fingerprint(typeInfo, frame + 8);
  Put64(frame + 8 + kFingerprintBytes, length);
  std::memcpy(frame + kLookupHeadBytes, name, length);
  bool sent{SendToPeer(peerHigh, peerLow, kLookup, frame,
      kLookupHeadBytes + padded, nullptr, 0)};
  std::free(frame);
  pthread_mutex_lock(&lock);
  std::int64_t status{3};
  if (sent) {
    for (;;) {
      if (pending.answered) {
        status = pending.status;
        break;
      }
      // Lost once no connection to the node is left to answer on.
      if (!CurrentOf(peerHigh, peerLow) &&
          !OtherOf(nullptr, peerHigh, peerLow, false)) {
        status = 4;
        break;
      }
      if (deadline >= 0 && Now() >= deadline) {
        status = 5;
        break;
      }
      WaitUntil(deadline);
    }
  }
  for (Pending **link{&pendings}; *link; link = &(*link)->next) {
    if (*link == &pending) {
      *link = pending.next;
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  if (status == 0) {
    *placeAndSlot = static_cast<std::int64_t>(pending.handle.placeAndSlot);
    *generation = pending.handle.generation;
    *secret = static_cast<std::int64_t>(pending.handle.secret);
  }
  return status;
}

// For tests: the connections that are not done.
extern "C" std::int64_t merk_node_connection_count() {
  pthread_mutex_lock(&lock);
  std::int64_t count{0};
  for (Connection *c{First()}; c; c = c->next) {
    count += c->done ? 0 : 1;
  }
  pthread_mutex_unlock(&lock);
  return count;
}
