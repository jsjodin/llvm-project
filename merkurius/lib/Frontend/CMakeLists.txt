add_merkurius_library(MerkuriusFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendActions.cpp

  DEPENDS
  MerkuriusDriverOptions
  MerkOpsIncGen
  MrtOpsIncGen

  LINK_LIBS
  MerkuriusCodeGen
  MerkuriusMerkDialect
  MerkuriusMrtDialect
  MerkuriusDriver
  MerkuriusParser
  MerkuriusSemantics
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRParser

  LINK_COMPONENTS
  BitWriter
  Core
  IRReader
  Option
  Support
  )
