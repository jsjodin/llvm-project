add_custom_target(MerkuriusUnitTests)
set_target_properties(MerkuriusUnitTests PROPERTIES FOLDER "Merkurius/Tests")

if(CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_merkurius_unittest test_dirname)
  add_unittest(MerkuriusUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Common)
