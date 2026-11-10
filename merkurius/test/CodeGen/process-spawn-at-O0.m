% process-spawn-at.m's program at -O0, whose GPU process image is linked at
% -O0 as -O asks. AMDGPU's code generation fails on it, at -O0 and -O1,
% with "failed to annotate CFG" on a musttail call in divergent control
% flow (design document, Driver, LLVM bugs to report). Expected to fail
% until LLVM is fixed: when it passes, move this run back into
% process-spawn-at.m.
%
% REQUIRES: amdgpu-registered-target, libomp
% XFAIL: *
% RUN: merkc -O0 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %S/process-spawn-at.m -o %t
% RUN: %libomp-env %t | FileCheck %S/process-spawn-at.m --match-full-lines --strict-whitespace
