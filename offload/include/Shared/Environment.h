//===-- Shared/Environment.h - OpenMP GPU environments ------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Environments shared between host and device.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_ENVIRONMENT_H
#define OMPTARGET_SHARED_ENVIRONMENT_H

#include <stdint.h>

struct IdentTy;

enum class DeviceDebugKind : uint32_t {
  Assertion = 1U << 0,
  FunctionTracing = 1U << 1,
  CommonIssues = 1U << 2,
  PGODump = 1U << 4,
};

/// LIBOMPTARGET_LOOP_BARRIER: whether a combined distribute and worksharing
/// loop on the device has a barrier across the team after each round. Unset,
/// each loop has it as the compiler asked; "on" and "off" override that for
/// every loop.
enum class LoopBarrierKind : uint32_t {
  AsCompiled = 0,
  On = 1,
  Off = 2,
};

struct DeviceEnvironmentTy {
  uint32_t DeviceDebugKind;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
  uint64_t ClockFrequency;
  uintptr_t IndirectCallTable;
  uint64_t IndirectCallTableSize;
  uint64_t HardwareParallelism;
  uint32_t LoopBarrier;
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
// Different from KernelEnvironmentTy below, this structure contains members
// that might be modified at runtime.
struct DynamicEnvironmentTy {
  /// Current indentation level for the function trace. Only accessed by thread
  /// 0.
  uint16_t DebugIndentionLevel;
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
struct ConfigurationEnvironmentTy {
  uint8_t UseGenericStateMachine = 2;
  uint8_t MayUseNestedParallelism = 2;
  uint8_t ExecMode = 0;
  // Information about (legal) launch configurations.
  //{
  int32_t MinThreads = -1;
  int32_t MaxThreads = -1;
  int32_t MinTeams = -1;
  int32_t MaxTeams = -1;
  int32_t ReductionDataSize = 0;
  //}
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
struct KernelEnvironmentTy {
  ConfigurationEnvironmentTy Configuration;
  IdentTy *Ident = nullptr;
  DynamicEnvironmentTy *DynamicEnv = nullptr;
};

/// The fallback types for the dynamic cgroup memory.
enum class DynCGroupMemFallbackType : uint8_t {
  /// None. Used for indicating that no fallback was triggered.
  None = 0,
  /// Abort the execution.
  Abort = None,
  /// Return null pointer.
  Null = 1,
  /// Allocate from a implementation defined memory space.
  DefaultMem = 2
};

struct KernelLaunchEnvironmentTy {
  void *ReductionBuffer = nullptr;
  void *DynCGroupMemFbPtr = nullptr;
  uint32_t ReductionTeamsDone = 0;
  uint32_t DynCGroupMemSize = 0;
  DynCGroupMemFallbackType DynCGroupMemFb = DynCGroupMemFallbackType::None;
};

#endif // OMPTARGET_SHARED_ENVIRONMENT_H
