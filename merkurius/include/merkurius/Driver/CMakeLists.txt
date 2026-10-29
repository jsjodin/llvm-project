set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_merkurius_tablegen_target(MerkuriusDriverOptions)
