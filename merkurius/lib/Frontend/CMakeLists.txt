add_merkurius_library(MerkuriusFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendActions.cpp

  DEPENDS
  MerkuriusDriverOptions

  LINK_LIBS
  MerkuriusDriver
  MerkuriusParser

  LINK_COMPONENTS
  Option
  Support
  )
