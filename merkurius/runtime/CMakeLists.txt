# libmerkrt, the runtime library linked into every program (design
# document, "Runtime system"). It is built as a static library in the build
# directory's lib, where the driver looks for it beside bin/merkc, and
# installed there too.
add_library(merkrt STATIC
  array.cpp
  exception.cpp
  heap.cpp
  main.cpp
  write.cpp
  )
set_target_properties(merkrt PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Merkurius/Runtime")
install(TARGETS merkrt
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT merkrt)
