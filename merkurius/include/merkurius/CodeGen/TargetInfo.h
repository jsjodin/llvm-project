//===-- include/merkurius/CodeGen/TargetInfo.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_TARGETINFO_H_
#define MERKURIUS_CODEGEN_TARGETINFO_H_

// What code generation knows of the target (design document, "Lowering and
// code generation", "Target information"): the triple, CPU and features, the
// LLVM data layout, and the address spaces. As in ClangIR, a module records
// all of it as attributes, so that lowering a module read back from its
// text needs no command line: Record writes them and FromModule reads them
// back. Unlike ClangIR, which uses Clang's TargetInfo, it is built on LLVM's
// TargetMachine alone, as Flang's frontend is.

#include "merkurius/CodeGen/TargetOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <string>

namespace merkurius::codegen {

// The target's numbers for the address spaces of Mercury code: the generic
// one, which every pointer converts to, and those of global,
// workgroup-shared, constant and per-thread private memory.
struct AddressSpaces {
  unsigned generic{0};
  unsigned global{0};
  unsigned shared{0};
  unsigned constant{0};
  unsigned private_{0};
};

class TargetInfo {
public:
  // The target the options name; null after setting `error` if LLVM has no
  // such target.
  static std::unique_ptr<TargetInfo> Create(
      const TargetOptions &, std::string &error);
  // The target a module records; null after setting `error` if it records
  // none, or one LLVM does not have.
  static std::unique_ptr<TargetInfo> FromModule(
      mlir::ModuleOp, std::string &error);

  // Records the target on the module: llvm.target_triple, llvm.data_layout
  // and its DLTI form, merk.target_cpu, merk.target_features,
  // merk.optimization_level and merk.address_spaces, and
  // merk.process_kernels for a compile of GPU processes' kernels.
  void Record(mlir::ModuleOp) const;
  // Whether a module that Record recorded is a compile of GPU processes'
  // kernels.
  static bool ProcessKernels(mlir::ModuleOp);

  const llvm::Triple &triple() const { return triple_; }
  const llvm::DataLayout &dataLayout() const { return dataLayout_; }
  llvm::TargetMachine &targetMachine() const { return *targetMachine_; }
  const TargetOptions &options() const { return options_; }
  const AddressSpaces &addressSpaces() const { return addressSpaces_; }

  // The address spaces the data layout gives for allocas, globals and code.
  unsigned allocaAddressSpace() const {
    return dataLayout_.getAllocaAddrSpace();
  }
  unsigned globalsAddressSpace() const {
    return dataLayout_.getDefaultGlobalsAddressSpace();
  }
  unsigned programAddressSpace() const {
    return dataLayout_.getProgramAddressSpace();
  }
  // The size of a pointer in an address space, and its ABI alignment, in
  // bits.
  unsigned PointerBits(unsigned addressSpace) const {
    return dataLayout_.getPointerSizeInBits(addressSpace);
  }
  unsigned PointerAlignBits(unsigned addressSpace) const {
    return dataLayout_.getPointerABIAlignment(addressSpace).value() * 8;
  }
  // The width of a word, which int, uint and float take: a generic
  // pointer's.
  unsigned WordBits() const { return PointerBits(addressSpaces_.generic); }
  // The tag bits a pointer to a cell of words has free: 3 for 8-byte words.
  unsigned TagBits() const;

  // What depends on the kind of target, as in ClangIR's TargetCIRGenInfo
  // and Clang's emitAMDGPUMetadata:
  // Whether procedures use LLVM's tailcc, whose musttail calls are
  // guaranteed. AMDGPU has no tailcc, but makes a musttail call with the C
  // convention a jump; procedures there use C's.
  bool HasTailCallConvention() const { return !triple_.isAMDGPU(); }
  // The registers a procedure's results are returned in, a word or a
  // float each, by the procedures' convention; with `shared`, floats are
  // in the integer registers too. A procedure whose results need more
  // returns them in memory (design document, "Calling convention"), since
  // a back end that returns them in memory itself cannot make a musttail
  // call of it.
  struct ResultRegisters {
    unsigned integers;
    unsigned floats;
    bool shared{false};
  };
  ResultRegisters Results() const;
  // The target's module flags: on AMDGPU, the code object version, and the
  // PIC level, which Flang's device compile gives with -pic-level 2.
  void AddModuleFlags(llvm::Module &) const;

private:
  TargetInfo() = default;
  static std::unique_ptr<TargetInfo> Make(
      const TargetOptions &, const std::string *dataLayout, std::string &error);

  TargetOptions options_;
  llvm::Triple triple_;
  std::unique_ptr<llvm::TargetMachine> targetMachine_;
  llvm::DataLayout dataLayout_{""};
  AddressSpaces addressSpaces_;
};

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_TARGETINFO_H_
