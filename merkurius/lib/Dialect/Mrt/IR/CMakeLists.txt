add_merkurius_library(MerkuriusMrtDialect
  MrtDialect.cpp

  DEPENDS
  MrtOpsIncGen

  LINK_LIBS
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces

  LINK_COMPONENTS
  Support
  )
