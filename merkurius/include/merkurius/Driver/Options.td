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

def fopenmp : Flag<["-"], "fopenmp">,
  HelpText<"Read %$omp directives">;
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
def fdebug_dump_types : Flag<["-"], "fdebug-dump-types">,
  HelpText<"Type check the source file and list each clause's variable "
           "types and the declaration chosen for each name">;
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

} // let Visibility = [FC1Option]

//===----------------------------------------------------------------------===//
// Driver only
//===----------------------------------------------------------------------===//

def _HASH_HASH_HASH : Flag<["-"], "###">,
  HelpText<"Print the jobs to run, without running them">;
def v : Flag<["-"], "v">,
  HelpText<"Show the jobs as they run">;
