set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_merkurius_tool(merkc
  driver.cpp
  fc1_main.cpp

  DEPENDS
  MerkuriusDriverOptions
  )

# merkc finds the standard library in the build directory.
add_dependencies(merkc merkurius-library)

target_link_libraries(merkc
  PRIVATE
  MerkuriusDriver
  MerkuriusFrontend
  MerkuriusFrontendTool
  MerkuriusParser
  )
