/* The floor under pingpong.m's rounds between nodes (README.md): two
   processes hand an 8-byte number back and forth over a TCP connection on
   loopback, with TCP_NODELAY as Merkurius's connections have
   (runtime/node.cpp), and a blocking read on each side, with nothing else:
   no encoding, routing or mailboxes.

     tcp-floor ROUNDS */

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int full(int fd, int64_t *value, int writing) {
  char *p = (char *)value;
  size_t left = sizeof *value;
  while (left > 0) {
    ssize_t n = writing ? write(fd, p, left) : read(fd, p, left);
    if (n <= 0) {
      return 0;
    }
    p += n;
    left -= (size_t)n;
  }
  return 1;
}

static void nodelay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

/* Answers each N with N + 1 until -1. */
static void echo(int fd) {
  int64_t n;
  while (full(fd, &n, 0) && n >= 0) {
    ++n;
    if (!full(fd, &n, 1)) {
      break;
    }
  }
}

/* Rounds rounds from n; the number of answers in order. */
static long exchange(int fd, int64_t n, long rounds) {
  long inOrder = 0;
  for (long i = 0; i < rounds; ++i) {
    int64_t m;
    if (!full(fd, &n, 1) || !full(fd, &m, 0)) {
      break;
    }
    inOrder += m == n + 1;
    n = m + 1;
  }
  return inOrder;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
  long rounds = argc > 1 ? atol(argv[1]) : 10000;
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in address = {0};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length = sizeof address;
  if (bind(listener, (struct sockaddr *)&address, sizeof address) != 0 ||
      listen(listener, 1) != 0 ||
      getsockname(listener, (struct sockaddr *)&address, &length) != 0) {
    perror("listen");
    return 1;
  }
  pid_t child = fork();
  if (child == 0) {
    int fd = accept(listener, NULL, NULL);
    nodelay(fd);
    echo(fd);
    return 0;
  }
  close(listener);
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (connect(fd, (struct sockaddr *)&address, sizeof address) != 0) {
    perror("connect");
    return 1;
  }
  nodelay(fd);
  exchange(fd, 0, 100);
  double t0 = now();
  long inOrder = exchange(fd, 200, rounds);
  double t1 = now();
  int64_t stop = -1;
  full(fd, &stop, 1);
  waitpid(child, NULL, 0);
  printf("%-10s %8ld rounds, %8ld in order, %10.2f us a round\n", "tcp",
      rounds, inOrder, rounds ? (t1 - t0) / rounds : 0.0);
  return inOrder == rounds ? 0 : 1;
}
