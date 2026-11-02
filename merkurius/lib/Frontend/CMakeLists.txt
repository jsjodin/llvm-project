add_merkurius_library(MerkuriusFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendActions.cpp

  DEPENDS
  MerkuriusDriverOptions
  MerkOpsIncGen

  LINK_LIBS
  MerkuriusCodeGen
  MerkuriusMerkDialect
  MerkuriusDriver
  MerkuriusParser
  MerkuriusSemantics
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser

  LINK_COMPONENTS
  Core
  Option
  Support
  )
