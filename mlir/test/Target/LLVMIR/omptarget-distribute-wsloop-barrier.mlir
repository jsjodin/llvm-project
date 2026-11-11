// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The flags argument of __kmpc_distribute_for_static_loop on the device: bit 0
// says each thread runs at most one iteration (spmd_no_loop), bit 1 asks for a
// barrier across the team after each round of iterations, which a combined
// distribute and worksharing loop has unless it is nowait, as clang's has.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @with_barrier(%x : i32) {
    omp.target kernel_type(spmd) host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.teams {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    llvm.return
  }
  llvm.func @nowait(%x : i32) {
    omp.target kernel_type(spmd) host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.teams {
        omp.parallel {
          omp.distribute {
            omp.wsloop nowait {
              omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    llvm.return
  }
  llvm.func @no_loop(%x : i32) {
    omp.target kernel_type(spmd_no_loop) host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.teams {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// CHECK: call void @__kmpc_distribute_for_static_loop_4u(ptr @{{.*}}, ptr @__omp_offloading_{{.*}}_with_barrier_l{{[0-9]+}}..omp_par, ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0, i32 0, i8 2)
// CHECK: call void @__kmpc_distribute_for_static_loop_4u(ptr @{{.*}}, ptr @__omp_offloading_{{.*}}_nowait_l{{[0-9]+}}..omp_par, ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0, i32 0, i8 0)
// CHECK: call void @__kmpc_distribute_for_static_loop_4u(ptr @{{.*}}, ptr @__omp_offloading_{{.*}}_no_loop_l{{[0-9]+}}..omp_par, ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0, i32 0, i8 1)
