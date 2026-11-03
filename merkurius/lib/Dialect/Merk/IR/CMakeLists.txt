add_merkurius_library(MerkuriusMerkDialect
  MerkDialect.cpp

  DEPENDS
  MerkOpsIncGen

  LINK_LIBS
  MLIRArithDialect
  MLIRCallInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces

  LINK_COMPONENTS
  Support
  )
