set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_merkurius_unittest(MerkuriusCodeGenTests
  OpenMPClausesTest.cpp
  )
