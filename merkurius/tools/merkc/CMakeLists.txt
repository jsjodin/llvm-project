set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Option
  Support
  )

add_merkurius_tool(merkc
  driver.cpp
  fc1_main.cpp

  DEPENDS
  MerkuriusDriverOptions
  )

# merkc finds the standard library in the build directory,
add_dependencies(merkc merkurius-library)
# and links programs with the runtime library beside it.
add_dependencies(merkc merkrt)

target_link_libraries(merkc
  PRIVATE
  MerkuriusDriver
  MerkuriusFrontend
  MerkuriusFrontendTool
  MerkuriusParser
  )
