add_merkurius_unittest(MerkuriusCommonTests
  EnumClassTest.cpp
  IndirectionTest.cpp
  VisitTest.cpp
  )

target_link_libraries(MerkuriusCommonTests
  PRIVATE
  MerkuriusCommon
  )
