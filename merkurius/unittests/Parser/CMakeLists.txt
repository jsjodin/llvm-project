add_merkurius_unittest(MerkuriusParserTests
  LexerTest.cpp
  OperatorsTest.cpp
  ProvenanceTest.cpp
  )

target_link_libraries(MerkuriusParserTests
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  )
