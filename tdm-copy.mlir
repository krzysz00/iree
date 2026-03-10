// RUN: iree-compile --output-format=vm-bytecode --iree-hal-target-backends=rocm --iree-rocm-target=gfx1250 --compile-from=executable-configurations -o tdm-matmul.vmfb tdm-matmul.mlir

/// Kernel that writes out the A and B inputs to the matrix multiplies we think we're doing
/// so we know that the TDM code is workind and applying padding correctly.

/// Note to future readers: the bug was that I had TDM calls on the wrong side of a barrier x.x.

#gpu_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx1250", features = "",
    wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup =  shuffle|arithmetic,
      subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 327680,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 128, simds_per_wgp = 4,
      vgpr_space_bits = 32768, workgroup_memory_bank_count = 64>>, ukernels = "none"}>

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [128, 1, 1] subgroup_size = 32,
        // Padding is done via TDM.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            no_reduce_shared_memory_bank_conflicts = true>}>

#compile_info = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{
        workgroup = [256, 256, 0]
      }>,
      translation_info = #translation_info>

#layout = #hal.pipeline.layout<constants = 8, bindings =
  [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
   #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
   #hal.pipeline.binding<storage_buffer, Indirect>,
   #hal.pipeline.binding<storage_buffer, Indirect>]>

!global_buffer_in = memref<?x?xi16, #gpu.address_space<global>>
!global_buffer_out = memref<?x?xi16, #gpu.address_space<global>>
!global_out_exp = memref<?x8x2x16x?x4x2x16xi16, #gpu.address_space<global>>
!tdm_tile = memref<256x136xi16, #gpu.address_space<workgroup>>
!lds_tile = memref<256x128xi16, strided<[136, 1]>, #gpu.address_space<workgroup>>
!matmul_in_tile = memref<8x2x16x4x2x16xi16, strided<[4352, 2176, 136, 32, 16, 1]>, #gpu.address_space<workgroup>>

module @test attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"hip", [#gpu_target]>
  hal.executable private @tdm_copy {
    hal.executable.variant public @rocm_hsaco_fb target(#gpu_target) {
      hal.executable.export public @tdm_copy ordinal(0) layout(#layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
        %0 = affine.min affine_map<()[s0, s1] -> (s0 * s1, 2147483647)>()[%arg3, %arg4]
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        hal.return %0, %c1, %c1_0 : index, index, index
      } attributes {subgroup_size = 32 : index, workgroup_size = [128 : index, 1 : index, 1 : index]}
      builtin.module {
        func.func @tdm_copy() attributes {translation_info = #translation_info} {
          %c32_i64 = arith.constant 32 : i64
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c128 = arith.constant 128 : index
          %c256 = arith.constant 256 : index
          %poison_i16 = ub.poison : i16

          %c64_i32 = arith.constant 64 : i32
          %c4_i32 = arith.constant 4 : i32

          %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
          %2 = hal.interface.constant.load layout(#layout) ordinal(2) : i32
          %3 = hal.interface.constant.load layout(#layout) ordinal(3) : i32
          %4 = hal.interface.constant.load layout(#layout) ordinal(4) : i32
          %5 = hal.interface.constant.load layout(#layout) ordinal(5) : i32
          %6 = hal.interface.constant.load layout(#layout) ordinal(6) : i32
          %7 = hal.interface.constant.load layout(#layout) ordinal(7) : i32
          %8 = arith.extui %0 : i32 to i64
          %9 = arith.extui %1 : i32 to i64
          %10 = arith.shli %9, %c32_i64 : i64
          %11 = arith.ori %8, %10 : i64
          %12 = arith.index_castui %11 : i64 to index
          %13 = arith.extui %2 : i32 to i64
          %14 = arith.extui %3 : i32 to i64
          %15 = arith.shli %14, %c32_i64 : i64
          %16 = arith.ori %13, %15 : i64
          %17 = arith.index_castui %16 : i64 to index
          %18 = arith.extui %4 : i32 to i64
          %19 = arith.extui %5 : i32 to i64
          %20 = arith.shli %19, %c32_i64 : i64
          %21 = arith.ori %18, %20 : i64
          %22 = arith.index_castui %21 : i64 to index
          %23 = arith.extui %6 : i32 to i64
          %24 = arith.extui %7 : i32 to i64
          %25 = arith.shli %24, %c32_i64 : i64
          %26 = arith.ori %23, %25 : i64
          %27 = arith.index_castui %26 : i64 to index
          %KL, %KR, %M, %N = util.assume.int
              %12<umin = 0, umax = 9007199254740991>,
              %17<umin = 0, umax = 9007199254740991>,
              %22<umin = 0, umax = 9007199254740991>,
              %27<umin = 0, umax = 9007199254740991>
            : index, index, index, index

          %KLp = affine.apply affine_map<()[s0] ->((s0 ceildiv 256) * 256)>()[%KL]
          %KRp = affine.apply affine_map<()[s0] ->((s0 ceildiv 256) * 256)>()[%KR]
          %Mp = affine.apply affine_map<()[s0] ->((s0 ceildiv 256) * 256)>()[%M]
          %Np = affine.apply affine_map<()[s0] ->((s0 ceildiv 256) * 256)>()[%N]
          %num_k = arith.divui %KLp, %c128 exact : index

          %A = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !global_buffer_in{%M, %KL}
          %B = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !global_buffer_in{%N, %KR}
          %A_out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !global_buffer_out{%Mp, %KLp}
          %B_out = hal.interface.binding.subspan layout(#layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !global_buffer_out{%Np, %KRp}

          %M_wg = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%M]
          %N_wg = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%N]
          %num_wg = arith.muli %M_wg, %N_wg overflow<nsw, nuw> : index

          %a0 = memref.alloc() : !tdm_tile
          %a1 = memref.alloc() : !tdm_tile
          %b0 = memref.alloc() : !tdm_tile
          %b1 = memref.alloc() : !tdm_tile

          %a0_nopad = memref.subview %a0[0, 0] [256, 128] [1, 1] : !tdm_tile to !lds_tile
          %a1_nopad = memref.subview %a1[0, 0] [256, 128] [1, 1] : !tdm_tile to !lds_tile
          %b0_nopad = memref.subview %b0[0, 0] [256, 128] [1, 1] : !tdm_tile to !lds_tile
          %b1_nopad = memref.subview %b1[0, 0] [256, 128] [1, 1] : !tdm_tile to !lds_tile

          %a0_exp = memref.expand_shape %a0_nopad [[0, 1, 2], [3, 4, 5]] output_shape [8, 2, 16, 4, 2, 16] : !lds_tile into !matmul_in_tile
          %a1_exp = memref.expand_shape %a1_nopad [[0, 1, 2], [3, 4, 5]] output_shape [8, 2, 16, 4, 2, 16] : !lds_tile into !matmul_in_tile
          %b0_exp = memref.expand_shape %b0_nopad [[0, 1, 2], [3, 4, 5]] output_shape [8, 2, 16, 4, 2, 16] : !lds_tile into !matmul_in_tile
          %b1_exp = memref.expand_shape %b1_nopad [[0, 1, 2], [3, 4, 5]] output_shape [8, 2, 16, 4, 2, 16] : !lds_tile into !matmul_in_tile

          %A_out_exp = memref.expand_shape %A_out [[0, 1, 2, 3], [4, 5, 6, 7]] output_shape [%M_wg, 8, 2, 16, %num_k, 4, 2, 16] : !global_buffer_out into !global_out_exp
          %B_out_exp = memref.expand_shape %B_out [[0, 1, 2, 3], [4, 5, 6, 7]] output_shape [%N_wg, 8, 2, 16, %num_k, 4, 2, 16] : !global_buffer_out into !global_out_exp

          pcf.loop scope(#iree_codegen.workgroup_scope) count (%num_wg)
              execute[%wgid : index] {
            %m_wg, %n_wg = affine.delinearize_index %wgid into (%M_wg, %N_wg) : index, index

            pcf.generic scope(#iree_gpu.subgroup_scope)
                execute[%sgid : index, %sg_count : index] {
              %m_global_off_load = affine.linearize_index disjoint [%m_wg, %sgid, %c0] by (4, 64) : index
              %n_global_off_load = affine.linearize_index disjoint [%n_wg, %sgid, %c0] by (4, 64) : index

              %m_tile_size_tmp = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%M, %m_global_off_load]
              %m_tile_size = affine.max affine_map<(d0)[] -> (d0, 0)>(%m_tile_size_tmp)
              %n_tile_size_tmp = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%N, %n_global_off_load]
              %n_tile_size = affine.max affine_map<(d0)[] -> (d0, 0)>(%n_tile_size_tmp)

              // Prologue.
              %k0_tile_size_prologue = affine.min affine_map<()[s0] -> (s0 - 0, 128)>(%KL)

              %lds_sg_off = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%sgid]

              %a0_base_prologue = amdgpu.make_dma_base %A[%m_global_off_load, %c0], %a0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
              %b0_base_prologue = amdgpu.make_dma_base %B[%n_global_off_load, %c0], %b0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
              %a0_desc_prologue = amdgpu.make_dma_descriptor %a0_base_prologue
                globalSize [%m_tile_size, %k0_tile_size_prologue]
                globalStride [%KL, 1]
                sharedSize [%c64, %c128]
                padShared (%c4_i32 every %c64_i32)
                : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>
              %b0_desc_prologue = amdgpu.make_dma_descriptor %b0_base_prologue
                globalSize [%n_tile_size, %k0_tile_size_prologue]
                globalStride [%KL, 1]
                sharedSize [%c64, %c128]
                padShared (%c4_i32 every %c64_i32)
                : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>
              amdgpu.tensor_load_to_lds %a0_desc_prologue : !amdgpu.tdm_descriptor<i16>
              amdgpu.tensor_load_to_lds %b0_desc_prologue : !amdgpu.tdm_descriptor<i16>
              // asyncmark goes here once we have it.

              %m_sg, %n_sg = affine.delinearize_index %sgid into (2, 2) : index, index

              // Compared to Triton, I unrolled the loop manually instead of
              // doing an explicit multi-buffer.
              pcf.generic scope(#iree_gpu.lane_scope)
                  execute [%laneid : index, %nlane : index] {
                %k_lane, %mn_lane = affine.delinearize_index %laneid into (2, 16) : index, index

                scf.for %k1 = %c128 to %KL step %c256 {
                  %k0 = arith.addi %k1, %c128 overflow<nsw, nuw> : index

                  // Load to buffer group 1.
                  %k1_tile_size_tmp = affine.min affine_map<()[s0, s1] ->(128, s0 - s1)>()[%KL, %k1]
                  %k1_tile_size = affine.max affine_map<()[s0] ->(0, s0)>()[%k1_tile_size_tmp]

                  %a1_base = amdgpu.make_dma_base %A[%m_global_off_load, %k1], %a1[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
                  %b1_base = amdgpu.make_dma_base %B[%n_global_off_load, %k1], %b1[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
                  %a1_desc = amdgpu.make_dma_descriptor %a1_base
                    globalSize [%m_tile_size, %k1_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>
                  %b1_desc = amdgpu.make_dma_descriptor %b1_base
                    globalSize [%n_tile_size, %k1_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>

                  // Barrier here so we don't clobber group 0.
                  amdgpu.memory_counter_wait tensor(2) // 2 reads per group
                  gpu.barrier memfence [#gpu.address_space<workgroup>]

                  amdgpu.tensor_load_to_lds %a1_desc : !amdgpu.tdm_descriptor<i16>
                  amdgpu.tensor_load_to_lds %b1_desc : !amdgpu.tdm_descriptor<i16>

                  // Compute on matrix group 0.
                  %lhs0_slice = vector.transfer_read %a0_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                  %lhs0 = vector.shape_cast %lhs0_slice : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>
                  %rhs0_slice = vector.transfer_read %b0_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                  %rhs0 = vector.shape_cast %rhs0_slice : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>
                  %k1_tile = arith.divui %k1, %c128 exact : index
                  %kPrev_tile = arith.subi %k1_tile, %c1 : index
                  vector.transfer_write %lhs0, %A_out_exp[%m_wg, %c0, %m_sg, %mn_lane, %kPrev_tile, %c0, %k_lane, %c0]
                    {in_bounds = [true, true, true, true, true, true, true, true]}
                    : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp
                  vector.transfer_write %rhs0, %B_out_exp[%n_wg, %c0, %n_sg, %mn_lane, %kPrev_tile, %c0, %k_lane, %c0]
                    {in_bounds = [true, true, true, true, true, true, true, true]}
                    : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp

                  // Load to buffer group 0.
                  %k0_tile_size_tmp = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%KL, %k0]
                  %k0_tile_size = affine.max affine_map<()[s0] ->(0, s0)>()[%k0_tile_size_tmp]

                  %a0_base = amdgpu.make_dma_base %A[%m_global_off_load, %k0], %a0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
                  %b0_base = amdgpu.make_dma_base %B[%n_global_off_load, %k0], %b0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<i16>
                  %a0_desc = amdgpu.make_dma_descriptor %a0_base
                    globalSize [%m_tile_size, %k0_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>
                  %b0_desc = amdgpu.make_dma_descriptor %b0_base
                    globalSize [%n_tile_size, %k0_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<i16> -> !amdgpu.tdm_descriptor<i16>

                  // Barriers here so we don't clobber group 0.
                  amdgpu.memory_counter_wait tensor(2) // 2 reads per group
                  gpu.barrier memfence [#gpu.address_space<workgroup>]

                  amdgpu.tensor_load_to_lds %a0_desc : !amdgpu.tdm_descriptor<i16>
                  amdgpu.tensor_load_to_lds %b0_desc : !amdgpu.tdm_descriptor<i16>

                  // Compute on matrix group 1.
                  %lhs1_slice = vector.transfer_read %a1_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                  %lhs1 = vector.shape_cast %lhs1_slice : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>
                  %rhs1_slice = vector.transfer_read %b1_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                  %rhs1 = vector.shape_cast %rhs1_slice : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>
                  vector.transfer_write %lhs1, %A_out_exp[%m_wg, %c0, %m_sg, %mn_lane, %k1_tile, %c0, %k_lane, %c0]
                    {in_bounds = [true, true, true, true, true, true, true, true]}
                    : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp
                  vector.transfer_write %rhs1, %B_out_exp[%n_wg, %c0, %n_sg, %mn_lane, %k1_tile, %c0, %k_lane, %c0]
                    {in_bounds = [true, true, true, true, true, true, true, true]}
                    : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp
                }

                // Epilogue.
                amdgpu.memory_counter_wait tensor(0)
                gpu.barrier memfence [#gpu.address_space<workgroup>]

                // Compute on matrix group 0.
                %lhs_slice_epilogue = vector.transfer_read %a0_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                %lhs_epilogue = vector.shape_cast %lhs_slice_epilogue : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>
                %rhs_slice_epilogue = vector.transfer_read %b0_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_i16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xi16>
                %rhs_epilogue = vector.shape_cast %rhs_slice_epilogue : vector<8x1x1x4x1x16xi16> to vector<1x8x1x1x1x4x1x16xi16>

                %kLast_tile = arith.subi %num_k, %c1 : index
                vector.transfer_write %lhs_epilogue, %A_out_exp[%m_wg, %c0, %m_sg, %mn_lane, %kLast_tile, %c0, %k_lane, %c0]
                  {in_bounds = [true, true, true, true, true, true, true, true]}
                  : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp
                vector.transfer_write %rhs_epilogue, %B_out_exp[%n_wg, %c0, %n_sg, %mn_lane, %kLast_tile, %c0, %k_lane, %c0]
                  {in_bounds = [true, true, true, true, true, true, true, true]}
                  : vector<1x8x1x1x1x4x1x16xi16>, !global_out_exp

                pcf.return
              }
              pcf.return
            }
            pcf.return
          }
          func.return
        }
      }
    }
  }
  util.func public @tdm_copy_wrap(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @tdm_matmul_trb_wrap(%input0: tensor<?x?xi16>, %input1: tensor<?x?xi16>) -> (%output0: tensor<?x?xi16>, %output1: tensor<?x?xi16>)"}} {
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_i16 = hal.element_type<i16> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_i16) encoding(%dense_row_major)
    %2 = arith.muli %0, %c2 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xi16>{%0, %1} in !stream.resource<external>{%3}
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%5, %6]) type(%element_type_i16) encoding(%dense_row_major)
    %7 = arith.muli %5, %c2 : index
    %8 = arith.muli %7, %6 : index
    %9 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?x?xi16>{%5, %6} in !stream.resource<external>{%8}
    %div0 = arith.ceildivui %0, %c256 : index
    %rd0 = arith.muli %div0, %c256 : index
    %div1 = arith.ceildivui %1, %c256 : index
    %rd1 = arith.muli %div1, %c256 : index
    %div5 = arith.ceildivui %5, %c256 : index
    %rd5 = arith.muli %div5, %c256 : index
    %div6 = arith.ceildivui %6, %c256 : index
    %rd6 = arith.muli %div6, %c256 : index

    %10 = arith.muli %rd0, %c2 : index
    %11 = arith.muli %10, %rd1 : index
    %b10 = arith.muli %rd5, %c2 : index
    %b11 = arith.muli %b10, %rd6 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%11} => !stream.timepoint
    %resultB, %result_timepointB = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => !stream.resource<external>{%b11} => !stream.timepoint

    %12 = arith.index_castui %1 : index to i64
    %13 = arith.index_castui %1 : index to i32
    %14 = arith.shrui %12, %c32_i64 : i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.index_castui %6 : index to i64
    %17 = arith.index_castui %6 : index to i32
    %18 = arith.shrui %16, %c32_i64 : i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.index_castui %0 : index to i64
    %21 = arith.index_castui %0 : index to i32
    %22 = arith.shrui %20, %c32_i64 : i64
    %23 = arith.trunci %22 : i64 to i32
    %24 = arith.index_castui %5 : index to i64
    %25 = arith.index_castui %5 : index to i32
    %26 = arith.shrui %24, %c32_i64 : i64
    %27 = arith.trunci %26 : i64 to i32
    %28 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepointB) => with(%4 as %arg2: !stream.resource<external>{%3}, %9 as %arg3: !stream.resource<external>{%8}, %result as %arg4: !stream.resource<external>{%11}, %resultB as %arg5 : !stream.resource<external>{%b11}) {
      stream.cmd.dispatch @tdm_copy::@rocm_hsaco_fb::@tdm_copy[%1, %6, %0, %5](%13, %15, %17, %19, %21, %23, %25, %27 : i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg2[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg3[%c0 for %8] : !stream.resource<external>{%8},
        wo %arg4[%c0 for %11] : !stream.resource<external>{%11},
        wo %arg5[%c0 for %b11] : !stream.resource<external>{%b11}
      }
    } => !stream.timepoint
    %29, %b29 = stream.timepoint.await %28 => %result, %resultB : !stream.resource<external>{%11}, !stream.resource<external>{%b11}
    %30 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %29 : tensor<?x?xi16>{%rd0, %rd1} in !stream.resource<external>{%11} -> !hal.buffer_view
    %b30 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %b29 : tensor<?x?xi16>{%rd5, %rd6} in !stream.resource<external>{%b11} -> !hal.buffer_view
    util.return %30, %b30 : !hal.buffer_view, !hal.buffer_view
  }
}
