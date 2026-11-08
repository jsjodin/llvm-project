// RUN: mlir-opt -split-input-file --omp-function-filter %s | FileCheck %s

// CHECK: llvm.func @any
// CHECK: llvm.return
// CHECK: llvm.func @nohost
// CHECK: llvm.return
// CHECK-NOT: llvm.func {{.*}}@host
// CHECK-NOT: llvm.func {{.*}}@none
// CHECK: llvm.func @nohost_target
// CHECK: llvm.return
// CHECK: llvm.func @host_target
// CHECK: llvm.return
// CHECK: llvm.func @none_target
// CHECK: llvm.return
// CHECK: llvm.func @host_target_call
// CHECK-NOT: llvm.call @none_target
// CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : i32
// CHECK: llvm.return %[[UNDEF]] : i32
module attributes {omp.is_target_device = true} {
  llvm.func @any() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = any, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = nohost, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = host, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @none() -> () {
    llvm.return
  }
  llvm.func @nohost_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = nohost, capture_clause = to>
      } {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    llvm.return
  }
  llvm.func @host_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = host, capture_clause = to>
      } {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    llvm.return
  }
  llvm.func @none_target() -> i32 {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    %0 = arith.constant 25 : i32
    llvm.return %0 : i32
  }
  llvm.func @host_target_call() -> i32
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = host, capture_clause = to>
      } {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    %0 = llvm.call @none_target() : () -> i32
    llvm.return %0 : i32
  }
}

// -----

// CHECK: llvm.func @any
// CHECK: llvm.return
// CHECK: llvm.func @nohost
// CHECK: llvm.return
// CHECK: llvm.func @host
// CHECK: llvm.return
// CHECK: llvm.func @none
// CHECK: llvm.return
// CHECK: llvm.func @nohost_target
// CHECK: llvm.return
// CHECK: llvm.func @host_target
// CHECK: llvm.return
// CHECK: llvm.func @none_target
// CHECK: llvm.return
module attributes {omp.is_target_device = false} {
  llvm.func @any() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = any, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @nohost() -> ()
      attributes {
          omp.declare_target =
            #omp.declaretarget<device_type = nohost, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = host, capture_clause = to>
      } {
    llvm.return
  }
  llvm.func @none() -> () {
    llvm.return
  }
  llvm.func @nohost_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = nohost, capture_clause = to>
      } {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    llvm.return
  }
  llvm.func @host_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = host, capture_clause = to>
      } {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    llvm.return
  }
  llvm.func @none_target() -> () {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    llvm.return
  }
}


// -----

// A result of a type that is not the LLVM dialect's, as before lowering to
// it, is replaced by ub.poison.

// CHECK-LABEL: func.func @index_target_call
// CHECK-NOT: call @host_index
// CHECK: %[[POISON:.*]] = ub.poison : index
// CHECK: return %[[POISON]] : index
// CHECK-NOT: func.func private @host_index
module attributes {omp.is_target_device = true} {
  func.func private @host_index() -> index
  func.func @index_target_call() -> index {
    omp.target kernel_type(generic) {
      omp.terminator
    }
    %0 = func.call @host_index() : () -> index
    return %0 : index
  }
}
