//===-- runtime/frame-info.h - Where a frame keeps its return address -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A tailcc function whose tail calls pass more stack arguments than it
// received reserves the room for them below its return address before it
// pushes the frame pointer, so its return address is not next to the saved
// frame pointer. The stack map does not say how much it reserves; the call
// frame information does (docs/INPROGRESS.md, Statepoints and stack maps).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_FRAME_INFO_H_
#define MERKURIUS_RUNTIME_FRAME_INFO_H_

#include <cstdint>

namespace merkurius::runtime {

// The bytes the frame of the function that returns to `returnAddress`
// reserves between its return address and its saved frame pointer, from the
// .eh_frame rule that gives its canonical frame address from the frame
// pointer, or -1 if the function has no such rule there.
std::int64_t ReserveAt(std::uint64_t returnAddress);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_FRAME_INFO_H_
