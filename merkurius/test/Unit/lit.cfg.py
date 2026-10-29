# -*- Python -*-

import os

import lit.formats

config.name = "Merkurius-Unit"
config.suffixes = []

# The unit-test binaries are found under the build's unittests directory.
config.test_source_root = os.path.join(config.merkurius_obj_root, "unittests")
config.test_exec_root = config.test_source_root

config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")
