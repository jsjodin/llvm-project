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
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  MLIRTransformUtils

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
