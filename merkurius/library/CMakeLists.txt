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

# libmerkstd: the modules that have an implementation so far, compiled by
# merkc for the host, as the driver links them into every program. Their
# foreign_procs are compiled by clang, so it is built first when it is part
# of the build.
set(compiled_modules io float math require pair maybe list solutions
  process string char array)
set(compiled_objects)
foreach(module ${compiled_modules})
  set(object ${CMAKE_CURRENT_BINARY_DIR}/${module}.o)
  set(depends merkc ${library_outputs})
  if(TARGET clang)
    list(APPEND depends clang)
  endif()
  # array's procedures are numbered by the sound stub's modes, which
  # programs see.
  set(source ${library_dir}/${module}.m)
  if(module STREQUAL "array")
    set(source ${sound_array})
  endif()
  add_custom_command(OUTPUT ${object}
    COMMAND merkc -c -O2 -fstub-unimplemented ${source} -o ${object}
    DEPENDS ${depends}
    COMMENT "Compiling the library module ${module}.m")
  list(APPEND compiled_objects ${object})
endforeach()
set_source_files_properties(${compiled_objects} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)

# The same modules in the merk dialect, compiled as for an OpenMP device, as
# MLIR bytecode in lib/merkurius/device: a device's compile copies the
# procedures its target regions reach from them (docs/INPROGRESS.md, Library
# code on the device). Their foreign code is still C text, which each
# target's compile gives to clang. Only AMD GPUs are offloaded to so far.
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(device_dir ${library_dir}/device)
  set(device_outputs)
  foreach(module ${compiled_modules})
    set(output ${device_dir}/${module}.merkbc)
    set(source ${library_dir}/${module}.m)
    if(module STREQUAL "array")
      set(source ${sound_array})
    endif()
    add_custom_command(OUTPUT ${output}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${device_dir}
      COMMAND merkc -fc1 -fopenmp -fopenmp-is-target-device
        -triple amdgcn-amd-amdhsa -fstub-unimplemented -emit-merk-bc
        ${source} -o ${output}
      DEPENDS merkc ${library_outputs}
      COMMENT "Compiling the library module ${module}.m for a device")
    list(APPEND device_outputs ${output})
  endforeach()
  add_custom_target(merkurius-device-library ALL DEPENDS ${device_outputs})
  set_target_properties(merkurius-device-library PROPERTIES
    FOLDER "Merkurius/Library")
  install(FILES ${device_outputs} DESTINATION lib/merkurius/device
    COMPONENT merkc)
endif()
add_library(merkstd STATIC ${compiled_objects})
set_target_properties(merkstd PROPERTIES
  LINKER_LANGUAGE C
  ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
  FOLDER "Merkurius/Library")
install(TARGETS merkstd
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT merkc)
