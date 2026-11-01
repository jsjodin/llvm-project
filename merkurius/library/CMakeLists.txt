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
# The sound array stub, which merkc loads before array.m unless
# -fmmc-uniqueness is given.
set(sound_array ${library_dir}/sound/array.m)
add_custom_command(OUTPUT ${sound_array}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${library_dir}/sound
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make-sound-array.py
    ${CMAKE_CURRENT_SOURCE_DIR}/array.m ${sound_array}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/array.m
    ${CMAKE_CURRENT_SOURCE_DIR}/make-sound-array.py
  COMMENT "Making the sound array stub")
list(APPEND library_outputs ${sound_array})
add_custom_target(merkurius-library ALL DEPENDS ${library_outputs})
set_target_properties(merkurius-library PROPERTIES FOLDER "Merkurius")
install(FILES ${library_sources} DESTINATION lib/merkurius
  COMPONENT merkc)
install(FILES ${sound_array} DESTINATION lib/merkurius/sound
  COMPONENT merkc)
