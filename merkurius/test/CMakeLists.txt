llvm_canonicalize_cmake_booleans(MERKURIUS_GTEST_AVAIL)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(MERKURIUS_TEST_DEPENDS merkc)
if(NOT MERKURIUS_STANDALONE_BUILD)
  list(APPEND MERKURIUS_TEST_DEPENDS FileCheck count not)
endif()

if(MERKURIUS_GTEST_AVAIL)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
    )
  list(APPEND MERKURIUS_TEST_DEPENDS MerkuriusUnitTests)
endif()

add_lit_testsuite(check-merkurius "Running the Merkurius regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MERKURIUS_TEST_DEPENDS}
  )
set_target_properties(check-merkurius PROPERTIES FOLDER "Merkurius/Tests")

add_lit_testsuites(MERKURIUS ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MERKURIUS_TEST_DEPENDS}
  )
