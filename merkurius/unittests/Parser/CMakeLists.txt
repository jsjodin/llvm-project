add_merkurius_unittest(MerkuriusParserTests
  ConvertTest.cpp
  LexerTest.cpp
  OperatorsTest.cpp
  ProvenanceTest.cpp
  TermReaderTest.cpp
  TermTest.cpp
  )

target_link_libraries(MerkuriusParserTests
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  )
