include(GNUInstallDirs)

macro(add_merkurius_tablegen_target target)
  add_public_tablegen_target(${target})
  add_dependencies(merkurius-headers ${target})
endmacro()

# add_merkurius_library(name sources...
#   [LINK_COMPONENTS llvm-components...] [LINK_LIBS libs...] [DEPENDS targets...])
function(add_merkurius_library name)
  llvm_add_library(${name} ${ARGN})
  add_dependencies(${name} merkurius-headers)
  if(BUILD_SHARED_LIBS OR NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(TARGETS ${name}
      COMPONENT ${name}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif()
  set_property(GLOBAL APPEND PROPERTY MERKURIUS_LIBS ${name})
  set_target_properties(${name} PROPERTIES FOLDER "Merkurius/Libraries")
endfunction()

macro(add_merkurius_tool name)
  add_llvm_executable(${name} ${ARGN})
  install(TARGETS ${name}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT ${name})
  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-${name} DEPENDS ${name} COMPONENT ${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Merkurius/Tools")
endmacro()
