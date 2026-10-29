set(LLVM_LINK_COMPONENTS
  Support
  )

# A test tool, not installed.
add_llvm_executable(merkurius-term-fuzzer
  merkurius-term-fuzzer.cpp
  )
set_target_properties(merkurius-term-fuzzer PROPERTIES FOLDER "Merkurius/Tools")

target_link_libraries(merkurius-term-fuzzer
  PRIVATE
  MerkuriusCommon
  MerkuriusParser
  )
