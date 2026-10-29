# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "Merkurius"
config.test_format = lit.formats.ShTest()

# Mercury sources, and .test files for tests that need no source of their own.
config.suffixes = [".m", ".test"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.merkurius_obj_root, "test")

# Inputs holds files that tests read. Unit is a separate suite, found through
# its lit.site.cfg.py in the build directory.
config.excludes = ["Inputs", "CMakeLists.txt"]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

tool_dirs = [config.merkurius_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["merkc"], tool_dirs)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
