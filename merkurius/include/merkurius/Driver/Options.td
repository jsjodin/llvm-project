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

} // let Visibility = [DefaultVis, FC1Option]

//===----------------------------------------------------------------------===//
// Frontend only
//===----------------------------------------------------------------------===//

let Visibility = [FC1Option] in {

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

} // let Visibility = [FC1Option]

//===----------------------------------------------------------------------===//
// Driver only
//===----------------------------------------------------------------------===//

def _HASH_HASH_HASH : Flag<["-"], "###">,
  HelpText<"Print the jobs to run, without running them">;
def v : Flag<["-"], "v">,
  HelpText<"Show the jobs as they run">;
