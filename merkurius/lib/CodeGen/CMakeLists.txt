add_merkurius_library(MerkuriusCodeGen
  Lowering.cpp
  MLIRGen.cpp
  TargetInfo.cpp

  DEPENDS
  MerkOpsIncGen

  LINK_LIBS
  MerkuriusMerkDialect
  MerkuriusParser
  MerkuriusSemantics
  MLIRBuiltinToLLVMIRTranslation
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Core
  IRReader
  Linker
  MC
  Passes
  Support
  Target
  TargetParser
  )
