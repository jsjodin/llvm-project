//===-- rpc-floor.cpp - A plain RPC round trip from a GPU ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The floor under pingpong.m's GPU rounds (README.md): a target region opens
// a port of the offload plugin's RPC server, LLVM libc's protocol, with an
// opcode of its own, sends a word and receives the host's answer, as
// Merkurius's device runtime does for each send and each poll, but with a
// callback that only adds one. The rounds are timed on the host, with an
// empty region's time, its launch, printed beside them.
//
//   rpc-floor ROUNDS
//
//===----------------------------------------------------------------------===//

#include "shared/rpc.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#pragma omp begin declare target
constexpr uint32_t floorOpcode{('m' << 24) | 0x7f};

// The OpenMP device runtime's client, which the plugin points at the host's
// buffer when it loads the image.
extern "C" [[gnu::weak]] rpc::Client __llvm_rpc_client;

static uint64_t RoundTrip(uint64_t x) {
#ifdef __AMDGPU__
  rpc::Client::Port port{__llvm_rpc_client.open<floorOpcode>()};
  port.send([=](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = x; });
  uint64_t answer{0};
  port.recv([&](rpc::Buffer *buffer, uint32_t) { answer = buffer->data[0]; });
  return answer;
#else
  return x + 1;
#endif
}
#pragma omp end declare target

extern "C" void __tgt_register_rpc_callback(unsigned (*)(void *, unsigned));

static unsigned AddOne(void *raw, unsigned) {
  auto &port{*static_cast<rpc::Server::Port *>(raw)};
  if (port.get_opcode() != floorOpcode) {
    return rpc::RPC_UNHANDLED_OPCODE;
  }
  port.recv_and_send([](rpc::Buffer *buffer, uint32_t) { ++buffer->data[0]; });
  return rpc::RPC_SUCCESS;
}

static double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static uint64_t Rounds(uint64_t rounds) {
  uint64_t x{0};
#pragma omp target map(tofrom : x)
  for (uint64_t i{0}; i < rounds; ++i) {
    x = RoundTrip(x);
  }
  return x;
}

int main(int argc, char **argv) {
  uint64_t rounds{argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 10000};
  __tgt_register_rpc_callback(AddOne);
  Rounds(100);
  double empty0{Now()};
  Rounds(0);
  double empty1{Now()};
  double t0{Now()};
  uint64_t x{Rounds(rounds)};
  double t1{Now()};
  std::printf("%-10s %8lu rounds, %8lu in order, %10.2f us a round\n",
      "rpc-floor", (unsigned long)rounds, (unsigned long)x,
      rounds ? (t1 - t0) / rounds : 0.0);
  std::printf("%-10s %10.2f us\n", "launch", empty1 - empty0);
  return x == rounds ? 0 : 1;
}
