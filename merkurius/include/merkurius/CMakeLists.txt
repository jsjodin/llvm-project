configure_file(Version.inc.in ${CMAKE_CURRENT_BINARY_DIR}/Version.inc)

add_subdirectory(Driver)
add_subdirectory(Dialect/Merk/IR)
add_subdirectory(Dialect/Mrt/IR)
