add_merkurius_library(MerkuriusFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  MerkuriusDriverOptions

  LINK_LIBS
  MerkuriusDriver
  MerkuriusFrontend

  LINK_COMPONENTS
  Option
  Support
  )
