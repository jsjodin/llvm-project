//===-- include/merkurius/Semantics/check-gpu-code.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_GPU_CODE_H_
#define MERKURIUS_SEMANTICS_CHECK_GPU_CODE_H_

// The checks by type of code that runs on a GPU (design document, "Messages
// and memory on the device"; docs/INPROGRESS.md, GPU processes without omp
// target, step 5a). A value is flat when its bytes say all of it, with no
// pointer: a scalar, an enumeration, a mailbox, a value of a type that is
// not recursive whose constructors' fields are such values (a flat record,
// or a tagged union of them), a tuple of them, a string, or an array whose
// elements are scalars or enumerations.
// - A message that code running on a GPU sends or receives is flat: in a
//   process kind declared `on gpu`, in a target region, and in the
//   procedures of the module that they call.
// - A GPU process kind's code, and the procedures of the module that it
//   calls, make only flat values, so that a GPU process allocates nothing
//   but arrays. A target region's code may make terms, which go when the
//   region ends.
// - A loop under distribute parallel do, which a GPU process runs across its
//   teams (docs/INPROGRESS.md, GPU processes that run loops), is in a GPU
//   process kind's code, or a procedure of the module that it calls; in a
//   target region, OpenMP's distribute parallel do is not implemented.
// Each is an error that names the type, at the goal. A procedure of another
// module is checked by the values it gives. It runs on the bodies that
// determinism analysis builds.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/insts.h"

namespace merkurius::semantics {

void CheckGpuCode(const parser::ParseTree &, const parser::TokenSequence &,
    const ModeChecking &, const DeterminismChecking &, InstTypes &constructors,
    parser::Messages &);

} // namespace merkurius::semantics

#endif // MERKURIUS_SEMANTICS_CHECK_GPU_CODE_H_
