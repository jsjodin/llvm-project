add_merkurius_unittest(MerkuriusParserTests
  LexerTest.cpp
  ProvenanceTest.cpp
  )

target_link_libraries(MerkuriusParserTests
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  )
