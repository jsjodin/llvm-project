//===-- LoongArchMCAsmInfo.cpp - LoongArch Asm properties ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the LoongArchMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "LoongArchMCAsmInfo.h"
#include "LoongArchMCExpr.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void LoongArchMCAsmInfo::anchor() {}

LoongArchMCAsmInfo::LoongArchMCAsmInfo(const Triple &TT) {
  CodePointerSize = CalleeSaveStackSlotSize = TT.isArch64Bit() ? 8 : 4;
  AlignmentIsInBytes = false;
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.dword\t";
  ZeroDirective = "\t.space\t";
  CommentString = "#";
  SupportsDebugInformation = true;
  DwarfRegNumForCFI = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
}

void LoongArchMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                            const MCSpecifierExpr &Expr) const {
  auto S = Expr.getSpecifier();
  bool HasSpecifier = S != 0 && S != ELF::R_LARCH_B26;
  if (HasSpecifier)
    OS << '%' << LoongArch::getSpecifierName(S) << '(';
  printExpr(OS, *Expr.getSubExpr());
  if (HasSpecifier)
    OS << ')';
}
