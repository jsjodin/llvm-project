/* The floor under pingpong.m's cpu-local rounds (README.md): two threads of
   one process hand a number back and forth through a mutex and a condition
   variable each way, as Merkurius's mailboxes wait (runtime/mailbox.cpp),
   with nothing else: no copying, encoding or queue.

     condvar-floor ROUNDS */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t toEcho = PTHREAD_COND_INITIALIZER;
static pthread_cond_t toMain = PTHREAD_COND_INITIALIZER;
static long ping = -2, pong = -2;

/* Answers each N with N + 1 until -1. */
static void *echo(void *unused) {
  (void)unused;
  pthread_mutex_lock(&lock);
  for (;;) {
    while (ping == -2) {
      pthread_cond_wait(&toEcho, &lock);
    }
    long n = ping;
    ping = -2;
    if (n < 0) {
      break;
    }
    pong = n + 1;
    pthread_cond_signal(&toMain);
  }
  pthread_mutex_unlock(&lock);
  return NULL;
}

/* Rounds rounds from n; the number of answers in order. */
static long exchange(long n, long rounds) {
  long inOrder = 0;
  pthread_mutex_lock(&lock);
  for (long i = 0; i < rounds; ++i) {
    ping = n;
    pthread_cond_signal(&toEcho);
    while (pong == -2) {
      pthread_cond_wait(&toMain, &lock);
    }
    inOrder += pong == n + 1;
    n = pong + 1;
    pong = -2;
  }
  pthread_mutex_unlock(&lock);
  return inOrder;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
  long rounds = argc > 1 ? atol(argv[1]) : 10000;
  pthread_t thread;
  pthread_create(&thread, NULL, echo, NULL);
  exchange(0, 100);
  double t0 = now();
  long inOrder = exchange(200, rounds);
  double t1 = now();
  pthread_mutex_lock(&lock);
  ping = -1;
  pthread_cond_signal(&toEcho);
  pthread_mutex_unlock(&lock);
  pthread_join(thread, NULL);
  printf("%-10s %8ld rounds, %8ld in order, %10.2f us a round\n",
      "condvar", rounds, inOrder, rounds ? (t1 - t0) / rounds : 0.0);
  return inOrder == rounds ? 0 : 1;
}
