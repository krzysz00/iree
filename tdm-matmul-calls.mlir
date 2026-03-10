// RUN: iree-compile --output-format=vm-bytecode --iree-hal-target-backends=rocm --iree-rocm-target=gfx1250 tdm-matmul-calls.mlir -o tdm-matmul-calls.vmfb
// RUN: ../build/tools/testing/e2e/iree-e2e-matmul-test --module=tdm-matmul.vmfb --module=tdm-matmul-calls.vmfb --device=hip
builtin.module @calls {

util.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
util.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

util.func private @module.tdm_matmul_trb_wrap(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

util.func @matmul_tdm_trb_1x1x1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %acc = util.null : !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 1 : i64
  %in_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 7 : i32
  %lhs = util.call @matmul_test.generate_random_matrix(%device, %m, %k, %in_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_seed = arith.constant 8 : i32
  %rhs = util.call @matmul_test.generate_random_matrix(%device, %n, %k, %in_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = util.call @module.tdm_matmul_trb_wrap(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %transpose_rhs = arith.constant 1 : i32
  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32,  !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  util.return
}

util.func @matmul_tdm_trb_256x256x256() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 256x256x256"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %acc = util.null : !hal.buffer_view
  %m = arith.constant 256 : i64
  %k = arith.constant 256 : i64
  %n = arith.constant 256 : i64
  %in_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 7 : i32
  %lhs = util.call @matmul_test.generate_random_matrix(%device, %m, %k, %in_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_seed = arith.constant 8 : i32
  %rhs = util.call @matmul_test.generate_random_matrix(%device, %n, %k, %in_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = util.call @module.tdm_matmul_trb_wrap(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %transpose_rhs = arith.constant 1 : i32
  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32,  !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  util.return
}

util.func @matmul_tdm_trb_33x1025x33() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 33x1025x33"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %acc = util.null : !hal.buffer_view
  %m = arith.constant 33 : i64
  %k = arith.constant 1025 : i64
  %n = arith.constant 33 : i64
  %in_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 7 : i32
  %lhs = util.call @matmul_test.generate_random_matrix(%device, %m, %k, %in_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_seed = arith.constant 8 : i32
  %rhs = util.call @matmul_test.generate_random_matrix(%device, %n, %k, %in_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = util.call @module.tdm_matmul_trb_wrap(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %transpose_rhs = arith.constant 1 : i32
  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32,  !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  util.return
}

util.func @matmul_tdm_trb_513x511x2() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 513x511x2"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %acc = util.null : !hal.buffer_view
  %m = arith.constant 513 : i64
  %k = arith.constant 2 : i64
  %n = arith.constant 511 : i64
  %in_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 7 : i32
  %lhs = util.call @matmul_test.generate_random_matrix(%device, %m, %k, %in_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_seed = arith.constant 8 : i32
  %rhs = util.call @matmul_test.generate_random_matrix(%device, %n, %k, %in_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = util.call @module.tdm_matmul_trb_wrap(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %transpose_rhs = arith.constant 1 : i32
  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32,  !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  util.return
}
}
