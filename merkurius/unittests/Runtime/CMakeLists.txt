add_merkurius_unittest(MerkuriusRuntimeTests
  EncodeTest.cpp
  MailboxTest.cpp
  )

target_include_directories(MerkuriusRuntimeTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime
  )

target_link_libraries(MerkuriusRuntimeTests
  PRIVATE
  merkrt
  )
