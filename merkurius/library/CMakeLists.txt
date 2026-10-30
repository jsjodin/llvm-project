# The standard library's modules, copied to lib/merkurius in the build
# directory, beside bin/merkc, where merkc looks for them, and installed
# there too.
file(GLOB library_sources CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.m)
set(library_dir ${CMAKE_BINARY_DIR}/lib/merkurius)
set(library_outputs)
foreach(source ${library_sources})
  get_filename_component(name ${source} NAME)
  add_custom_command(OUTPUT ${library_dir}/${name}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source}
      ${library_dir}/${name}
    DEPENDS ${source}
    COMMENT "Copying the library module ${name}")
  list(APPEND library_outputs ${library_dir}/${name})
endforeach()
add_custom_target(merkurius-library ALL DEPENDS ${library_outputs})
set_target_properties(merkurius-library PROPERTIES FOLDER "Merkurius")
install(FILES ${library_sources} DESTINATION lib/merkurius
  COMPONENT merkc)
