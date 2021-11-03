# Lint as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Bazel to CMake target name conversions used by bazel_to_cmake.py.

EXPLICIT_TARGET_MAPPING = {
    # Internal utilities to emulate various binary/library options.
    "//build_tools:default_linkopts": [],
    "//build_tools:dl": ["${CMAKE_DL_LIBS}"],

    # IREE llvm-external-projects
    "//llvm-external-projects/iree-dialects:IREEDialect": [
        "IREEDialectsIREEDialect"
    ],

    # LLVM
    "@llvm-project//llvm:IPO": ["LLVMipo"],
    "@llvm-project//lld:lld": ["lld"],
    # MLIR
    "@llvm-project//mlir:AllPassesAndDialects": ["MLIRAllDialects"],
    "@llvm-project//mlir:AffineToStandardTransforms": ["MLIRAffineToStandard"],
    "@llvm-project//mlir:CFGTransforms": ["MLIRSCFToStandard"],
    "@llvm-project//mlir:ComplexDialect": ["MLIRComplex"],
    "@llvm-project//mlir:DialectUtils": [""],
    "@llvm-project//mlir:ExecutionEngineUtils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:GPUDialect": ["MLIRGPUOps"],
    "@llvm-project//mlir:GPUTransforms": ["MLIRGPUTransforms"],
    "@llvm-project//mlir:LinalgInterfaces": ["MLIRLinalg"],
    "@llvm-project//mlir:LinalgOps": ["MLIRLinalg"],
    "@llvm-project//mlir:LLVMDialect": ["MLIRLLVMIR"],
    "@llvm-project//mlir:LLVMTransforms": ["MLIRStandardToLLVM"],
    "@llvm-project//mlir:MathDialect": ["MLIRMath"],
    "@llvm-project//mlir:ArithmeticDialect": ["MLIRArithmetic"],
    "@llvm-project//mlir:MemRefDialect": ["MLIRMemRef"],
    "@llvm-project//mlir:SCFToGPUPass": ["MLIRSCFToGPU"],
    "@llvm-project//mlir:SCFDialect": ["MLIRSCF"],
    "@llvm-project//mlir:StandardOps": ["MLIRStandard"],
    "@llvm-project//mlir:ShapeTransforms": ["MLIRShapeOpsTransforms"],
    "@llvm-project//mlir:SideEffects": ["MLIRSideEffectInterfaces"],
    "@llvm-project//mlir:SPIRVDialect": ["MLIRSPIRV"],
    "@llvm-project//mlir:TosaDialect": ["MLIRTosa"],
    "@llvm-project//mlir:ToLLVMIRTranslation": ["MLIRTargetLLVMIRExport"],
    "@llvm-project//mlir:mlir_c_runner_utils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:mlir-translate": ["mlir-translate"],
    "@llvm-project//mlir:MlirTableGenMain": ["MLIRTableGen"],
    "@llvm-project//mlir:MlirOptLib": ["MLIROptLib"],
    "@llvm-project//mlir:VectorOps": ["MLIRVector"],
    "@llvm-project//mlir:TensorDialect": ["MLIRTensor"],
    "@llvm-project//mlir:NVVMDialect": ["MLIRNVVMIR"],
    "@llvm-project//mlir:ROCDLDialect": ["MLIRROCDLIR"],
    # Vulkan
    "@vulkan_headers": ["Vulkan::Headers"],
    # Cuda
    "@cuda//:cuda_headers": ["cuda_headers"],
    # The Bazel target maps to the IMPORTED target defined by FindVulkan().
    "@vulkan_sdk//:sdk": ["Vulkan::Vulkan"],
    # Misc single targets
    "@com_google_benchmark//:benchmark": ["benchmark"],
    "@com_github_dvidelabs_flatcc//:flatcc": ["flatcc"],
    "@com_github_dvidelabs_flatcc//:parsing": ["flatcc::parsing"],
    "@com_github_dvidelabs_flatcc//:runtime": ["flatcc::runtime"],
    "@com_github_yaml_libyaml//:yaml": ["yaml"],
    "@com_google_googletest//:gtest": ["gmock", "gtest"],
    "@spirv_cross//:spirv_cross_lib": ["spirv-cross-msl"],
    "@cpuinfo": ["cpuinfo"],
    "@vulkan_memory_allocator//:impl_header_only": ["vulkan_memory_allocator"],
}


def _convert_mlir_target(target):
  # Default to a pattern substitution approach.
  # Take "MLIR" and append the name part of the full target identifier, e.g.
  #   "@llvm-project//mlir:IR"   -> "MLIRIR"
  #   "@llvm-project//mlir:Pass" -> "MLIRPass"
  return ["MLIR" + target.rsplit(":")[-1]]


def _convert_llvm_target(target):
  # Default to a pattern substitution approach.
  # Prepend "LLVM" to the Bazel target name.
  #   "@llvm-project//llvm:AsmParser" -> "LLVMAsmParser"
  #   "@llvm-project//llvm:Core" -> "LLVMCore"
  return ["LLVM" + target.rsplit(":")[-1]]


def convert_target(target):
  """Converts a Bazel target to a list of CMake targets.

  IREE targets are expected to follow a standard form between Bazel and CMake
  that facilitates conversion. External targets *may* have their own patterns,
  or they may be purely special cases.

  Multiple target in Bazel may map to a single target in CMake and a Bazel
  target may map to multiple CMake targets.

  Returns:
    A list of converted targets if it was successfully converted.

  Raises:
    KeyError: No conversion was found for the target.
  """
  if target in EXPLICIT_TARGET_MAPPING:
    return EXPLICIT_TARGET_MAPPING[target]
  if not target.startswith("@"):
    # Bazel `:api`            -> CMake `::api`
    # Bazel `//iree/base`     -> CMake `iree::base`
    # Bazel `//iree/base:foo` -> CMake `iree::base::foo`
    if target.startswith("//"):
      target = target[len("//"):]
    target = target.replace(":", "::")  # iree/base::foo or ::foo
    target = target.replace("/", "::")  # iree::base
    return [target]
  if target.startswith("@llvm-project//llvm"):
    return _convert_llvm_target(target)
  if target.startswith("@llvm-project//mlir"):
    return _convert_mlir_target(target)
  if target.startswith("@mlir-hlo//"):
    # All Bazel targets map to a single CMake target.
    return ["tensorflow::mlir_hlo"]

  raise KeyError(f"No conversion found for target '{target}'")
