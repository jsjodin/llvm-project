add_merkurius_unittest(MerkuriusSemanticsTests
  DesugarTest.cpp
  SymbolsTest.cpp
  TypesTest.cpp
  )

target_link_libraries(MerkuriusSemanticsTests
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  MerkuriusSemantics
  )
