//===-- include/merkurius/Driver/Options.td - merkc options -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for the driver (merkc) and the frontend (merkc -fc1). As in Clang,
// each option is marked with the layers that accept it: DefaultVis for the
// driver, FC1Option for the frontend.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def FC1Option : OptionVisibility;

//===----------------------------------------------------------------------===//
// Driver and frontend
//===----------------------------------------------------------------------===//

let Visibility = [DefaultVis, FC1Option] in {

def help : Flag<["-", "--"], "help">,
  HelpText<"Display available options">;
def version : Flag<["--"], "version">,
  HelpText<"Print version information">;

def E : Flag<["-"], "E">,
  HelpText<"Print the source as the lexer reads it">;
def o : JoinedOrSeparate<["-"], "o">, MetaVarName<"<file>">,
  HelpText<"Write output to <file>">;
def S : Flag<["-"], "S">,
  HelpText<"Compile to the target's assembly">;
def O : Joined<["-"], "O">, MetaVarName<"<level>">,
  HelpText<"Optimization level, 0 to 3">;

def fopenmp : Flag<["-"], "fopenmp">,
  HelpText<"Read %$omp directives">;
def fdo_loops : Flag<["-"], "fdo-loops">,
  HelpText<"Read ( Iterators do Body ) loops, which are not Mercury">;
def fopenmp_version_EQ : Joined<["-"], "fopenmp-version=">,
  MetaVarName<"<version>">,
  HelpText<"OpenMP version to accept, such as 52 for OpenMP 5.2">;
def I : JoinedOrSeparate<["-"], "I">, MetaVarName<"<dir>">,
  HelpText<"Add <dir> to the directories searched for imported modules">;

} // let Visibility = [DefaultVis, FC1Option]

//===----------------------------------------------------------------------===//
// Frontend only
//===----------------------------------------------------------------------===//

let Visibility = [FC1Option] in {

def library_dir : Separate<["-"], "library-dir">, MetaVarName<"<dir>">,
  HelpText<"The directory of the standard library's modules; by default "
           "lib/merkurius beside the directory of merkc">;
def fno_infer_types : Flag<["-"], "fno-infer-types">,
  HelpText<"Report each predicate or function without a type declaration, "
           "as mmc does without --infer-types; its type is still inferred">;
def ftype_inference_iteration_limit_EQ :
  Joined<["-"], "ftype-inference-iteration-limit=">, MetaVarName<"<n>">,
  HelpText<"Perform at most <n> passes of type inference (default: 60)">;
def fdebug_dump_types : Flag<["-"], "fdebug-dump-types">,
  HelpText<"Type check the source file and list each clause's variable "
           "types and the declaration chosen for each name">;
def finfer_modes : Flag<["-"], "finfer-modes">,
  HelpText<"Infer the modes of predicates without mode declarations, as "
           "mmc's --infer-modes does, rather than report each">;
def fmmc_uniqueness : Flag<["-"], "fmmc-uniqueness">,
  HelpText<"Find the unique values needed on backtracking as mmc does, "
           "which misses some that an if-then-else needs, keep a unique "
           "value put in a term unique while both are used, and compare the "
           "uniqueness of copied values, such as numbers, as mmc does">;
def fmode_inference_iteration_limit_EQ :
  Joined<["-"], "fmode-inference-iteration-limit=">, MetaVarName<"<n>">,
  HelpText<"Perform at most <n> passes of mode inference (default: 30)">;
def fdebug_dump_loops : Flag<["-"], "fdebug-dump-loops">,
  HelpText<"Type and purity check the source file and list each clause "
           "with loops as translated, and its loops' predicates">;
def fdebug_dump_modes : Flag<["-"], "fdebug-dump-modes">,
  HelpText<"Type, purity and mode check the source file and list each "
           "predicate's and function's procedures">;
def fdebug_dump_determinism : Flag<["-"], "fdebug-dump-determinism">,
  HelpText<"Type, purity and mode check the source file and list each "
           "procedure's body with its switches">;
def fdebug_dump_polymorphism : Flag<["-"], "fdebug-dump-polymorphism">,
  HelpText<"Check the source file and list where each procedure's "
           "type_infos come from">;
def fdebug_dump_resolved_names : Flag<["-"], "fdebug-dump-resolved-names">,
  HelpText<"Resolve the names of the source file and list the module's "
           "symbols and what each name resolves to">;
def fdebug_dump_imports : Flag<["-"], "fdebug-dump-imports">,
  HelpText<"Load the modules the source file imports and list each one's "
           "interface">;

def fdebug_dump_provenance : Flag<["-"], "fdebug-dump-provenance">,
  HelpText<"Dump the provenance table of the source files">;
def fdebug_dump_tokens : Flag<["-"], "fdebug-dump-tokens">,
  HelpText<"Dump the tokens of the source file">;
def fdebug_unparse_tokens : Flag<["-"], "fdebug-unparse-tokens">,
  HelpText<"Print the source file from its tokens and their trivia">;
def fdebug_dump_terms : Flag<["-"], "fdebug-dump-terms">,
  HelpText<"Print each item's term in canonical notation">;
def fdebug_dump_parse_tree_no_sema : Flag<["-"], "fdebug-dump-parse-tree-no-sema">,
  HelpText<"Print the parse tree, before semantic analysis">;
def fdebug_dump_semantic_form : Flag<["-"], "fdebug-dump-semantic-form">,
  HelpText<"Print the semantic form of each clause, after desugaring">;
def fdebug_dump_nonlocals : Flag<["-"], "fdebug-dump-nonlocals">,
  HelpText<"Print the semantic form as -fdebug-dump-semantic-form does, "
           "with each goal's nonlocal variables">;
def fdebug_unparse_exact : Flag<["-"], "fdebug-unparse-exact">,
  HelpText<"Print the parse tree as the source it was read from, checking "
           "that the tree accounts for every token">;
def fdebug_unparse_exact_minimal_layout
    : Flag<["-"], "fdebug-unparse-exact-minimal-layout">,
  HelpText<"Print the parse tree as -fdebug-unparse-exact does, without "
           "comments and with minimal layout">;
def fdebug_unparse_no_sema : Flag<["-"], "fdebug-unparse-no-sema">,
  HelpText<"Print the parse tree as source in the canonical layout, before "
           "semantic analysis">;
def fdebug_unparse_parenthesized_no_sema
    : Flag<["-"], "fdebug-unparse-parenthesized-no-sema">,
  HelpText<"Print the parse tree as -fdebug-unparse-no-sema does, with every "
           "operator term in parentheses">;

def triple : Separate<["-"], "triple">, MetaVarName<"<triple>">,
  HelpText<"Compile for the target <triple>; by default the host">;
def target_cpu : Separate<["-"], "target-cpu">, MetaVarName<"<cpu>">,
  HelpText<"Compile for the CPU <cpu>">;
def target_feature : Separate<["-"], "target-feature">,
  MetaVarName<"<feature>">,
  HelpText<"Turn a target feature on (+name) or off (-name)">;
def fopenmp_is_target_device : Flag<["-"], "fopenmp-is-target-device">,
  HelpText<"Compile the module's target regions for an OpenMP device">;
def fopenmp_host_ir_file_path : Separate<["-"], "fopenmp-host-ir-file-path">,
  MetaVarName<"<file>">,
  HelpText<"The host's LLVM IR, whose offload entries the device compile "
           "matches">;
def offload_targets_EQ : CommaJoined<["--"], "offload-targets=">,
  MetaVarName<"<triples>">,
  HelpText<"The OpenMP device triples the host compile makes offload entries "
           "for">;
def fopenmp_targets_EQ : CommaJoined<["-"], "fopenmp-targets=">,
  Alias<offload_targets_EQ>, MetaVarName<"<triples>">,
  HelpText<"The same as --offload-targets=">;
def fembed_offload_object_EQ : Joined<["-"], "fembed-offload-object=">,
  MetaVarName<"<file>">,
  HelpText<"Embed the offload binary <file> in the host's object file">;
def x : Separate<["-"], "x">, MetaVarName<"<language>">,
  HelpText<"Read the input as <language>: mercury, mlir or ir (LLVM IR); by "
           "default from its extension">;
def emit_merk : Flag<["-"], "emit-merk">,
  HelpText<"Write the module in the merk dialect">;
def emit_mlir : Flag<["-"], "emit-mlir">,
  HelpText<"Write the module lowered to MLIR's LLVM dialect">;
def emit_llvm : Flag<["-"], "emit-llvm">,
  HelpText<"Write the module's LLVM IR">;
def emit_llvm_bc : Flag<["-"], "emit-llvm-bc">,
  HelpText<"Write the module's LLVM bitcode">;
def emit_obj : Flag<["-"], "emit-obj">,
  HelpText<"Write the module's object file">;

} // let Visibility = [FC1Option]

//===----------------------------------------------------------------------===//
// Driver only
//===----------------------------------------------------------------------===//

def c : Flag<["-"], "c">,
  HelpText<"Compile each module to an object file, without linking">;
def target_EQ : Joined<["--"], "target=">, MetaVarName<"<triple>">,
  HelpText<"Compile and link for the target <triple>">;
def offload_arch_EQ : CommaJoined<["--"], "offload-arch=">,
  MetaVarName<"<gpus>">,
  HelpText<"With -fopenmp, compile the target regions for the GPUs <gpus> "
           "too, such as gfx1030, and link for offloading to them">;
def L : JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Search <dir> for libraries when linking">;

def _HASH_HASH_HASH : Flag<["-"], "###">,
  HelpText<"Print the jobs to run, without running them">;
def v : Flag<["-"], "v">,
  HelpText<"Show the jobs as they run">;
