add_merkurius_unittest(MerkuriusParserTests
  LexerTest.cpp
  OperatorsTest.cpp
  ProvenanceTest.cpp
  TermTest.cpp
  )

target_link_libraries(MerkuriusParserTests
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  )
