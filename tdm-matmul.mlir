// RUN: iree-compile --output-format=vm-bytecode --iree-hal-target-backends=rocm --iree-rocm-target=gfx1250 --compile-from=executable-configurations -o tdm-matmul.vmfb tdm-matmul.mlir

/// A quick prototype TDM code based on Triton.
/// Uses a 256x256x128 workgroup tile across four waves,
/// calling a 16x16x32 intrinsic. Double-buffers in LDS.

#gpu_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx1250", features = "",
    wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup =  shuffle|arithmetic,
      mma = [<WMMA_F32_16x16x4_F32>, <WMMA_F32_16x16x32_F16>, <WMMA_F32_16x16x32_BF16>, <WMMA_F16_16x16x32_F16>, <WMMA_BF16_16x16x32_BF16>, <WMMA_F32_16x16x64_F8E4M3FN>, <WMMA_F32_16x16x64_F8E4M3FN_F8E5M2>, <WMMA_F32_16x16x64_F8E5M2>, <WMMA_F32_16x16x64_F8E5M2_F8E4M3FN>, <WMMA_F16_16x16x64_F8E4M3FN>, <WMMA_F16_16x16x64_F8E4M3FN_F8E5M2>, <WMMA_F16_16x16x64_F8E5M2>, <WMMA_F16_16x16x64_F8E5M2_F8E4M3FN>, <WMMA_I32_16x16x64_I8>, <WMMA_F32_16x16x128_F8E5M2>, <WMMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <WMMA_F32_16x16x128_F8E4M3FN>, <WMMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <WMMA_F16_16x16x128_F8E5M2>, <WMMA_F16_16x16x128_F8E5M2_F8E4M3FN>, <WMMA_F16_16x16x128_F8E4M3FN>, <WMMA_F16_16x16x128_F8E4M3FN_F8E5M2>],
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
   #hal.pipeline.binding<storage_buffer, Indirect>]>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

!global_buffer_in = memref<?x?xf16, #gpu.address_space<global>>
!global_buffer_out = memref<?x?xf32, #gpu.address_space<global>>
!tdm_tile = memref<256x136xf16, #gpu.address_space<workgroup>>
!lds_tile = memref<256x128xf16, strided<[136, 1]>, #gpu.address_space<workgroup>>
!matmul_in_tile = memref<8x2x16x4x2x16xf16, strided<[4352, 2176, 136, 32, 16, 1]>, #gpu.address_space<workgroup>>

module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"hip", [#gpu_target]>
  hal.executable private @tdm_matmul_trb {
    hal.executable.variant public @rocm_hsaco_fb target(#gpu_target) {
      hal.executable.export public @tdm_matmul_trb ordinal(0) layout(#layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
        %0 = affine.min affine_map<()[s0, s1] -> (s0 * s1, 2147483647)>()[%arg3, %arg4]
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        hal.return %0, %c1, %c1_0 : index, index, index
      } attributes {subgroup_size = 32 : index, workgroup_size = [128 : index, 1 : index, 1 : index]}
      builtin.module {
        func.func @tdm_matmul_trb() attributes {translation_info = #translation_info} {
          %c32_i64 = arith.constant 32 : i64
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c128 = arith.constant 128 : index
          %c256 = arith.constant 256 : index
          %poison_f16 = ub.poison : f16

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

          %A = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !global_buffer_in{%M, %KL}
          %B = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !global_buffer_in{%N, %KR}
          %C = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !global_buffer_out{%M, %N}

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

              %a0_base_prologue = amdgpu.make_dma_base %A[%m_global_off_load, %c0], %a0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
              %b0_base_prologue = amdgpu.make_dma_base %B[%n_global_off_load, %c0], %b0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
              %a0_desc_prologue = amdgpu.make_dma_descriptor %a0_base_prologue
                globalSize [%m_tile_size, %k0_tile_size_prologue]
                globalStride [%KL, 1]
                sharedSize [%c64, %c128]
                padShared (%c4_i32 every %c64_i32)
                : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>
              %b0_desc_prologue = amdgpu.make_dma_descriptor %b0_base_prologue
                globalSize [%n_tile_size, %k0_tile_size_prologue]
                globalStride [%KL, 1]
                sharedSize [%c64, %c128]
                padShared (%c4_i32 every %c64_i32)
                : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>
              amdgpu.tensor_load_to_lds %a0_desc_prologue : !amdgpu.tdm_descriptor<f16>
              amdgpu.tensor_load_to_lds %b0_desc_prologue : !amdgpu.tdm_descriptor<f16>
              // asyncmark goes here once we have it.

              %m_sg, %n_sg = affine.delinearize_index %sgid into (2, 2) : index, index

              // Compared to Triton, I unrolled the loop manually instead of
              // doing an explicit multi-buffer.
              pcf.generic scope(#iree_gpu.lane_scope)
                  execute [%laneid : index, %nlane : index] {
                %accZeros = arith.constant dense<0.0> : vector<8x8x1x8xf32>
                %k_lane, %mn_lane = affine.delinearize_index %laneid into (2, 16) : index, index

                %accLoop = scf.for %k1 = %c128 to %KL step %c256 iter_args (%acc = %accZeros) -> (vector<8x8x1x8xf32>) {
                  %k0 = arith.addi %k1, %c128 overflow<nsw, nuw> : index

                  // Load to buffer group 1.
                  %k1_tile_size_tmp = affine.min affine_map<()[s0, s1] ->(128, s0 - s1)>()[%KL, %k1]
                  %k1_tile_size = affine.max affine_map<()[s0] ->(0, s0)>()[%k1_tile_size_tmp]

                  %a1_base = amdgpu.make_dma_base %A[%m_global_off_load, %k1], %a1[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
                  %b1_base = amdgpu.make_dma_base %B[%n_global_off_load, %k1], %b1[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
                  %a1_desc = amdgpu.make_dma_descriptor %a1_base
                    globalSize [%m_tile_size, %k1_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>
                  %b1_desc = amdgpu.make_dma_descriptor %b1_base
                    globalSize [%n_tile_size, %k1_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>

                  // Barrier here so we don't clobber group 0.
                  amdgpu.memory_counter_wait tensor(2) // 2 reads per group
                  gpu.barrier memfence [#gpu.address_space<workgroup>]

                  amdgpu.tensor_load_to_lds %a1_desc : !amdgpu.tdm_descriptor<f16>
                  amdgpu.tensor_load_to_lds %b1_desc : !amdgpu.tdm_descriptor<f16>

                  // Compute on matrix group 0.
                  %lhs0_slice = vector.transfer_read %a0_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                  %lhs0 = vector.shape_cast %lhs0_slice : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>
                  %rhs0_slice = vector.transfer_read %b0_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                  %rhs0 = vector.shape_cast %rhs0_slice : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>

                  // NOTE: Here and below, we use elements of B as the "lhs" so we
                  // can get wide writes below. I know we support something like this
                  // for MFMA, but it isn't plumbed through for WMMA.
                  // TODO: swap this for the col_major flag.
                  %acc0 = iree_codegen.inner_tiled ins(%rhs0, %lhs0) outs(%acc)
                    {
                      indexing_maps = #contraction_accesses,
                      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
                      kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
                    } : vector<8x4x1x16xf16>, vector<8x4x1x16xf16> into vector<8x8x1x8xf32>

                  // Load to buffer group 0.
                  %k0_tile_size_tmp = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%KL, %k0]
                  %k0_tile_size = affine.max affine_map<()[s0] ->(0, s0)>()[%k0_tile_size_tmp]

                  %a0_base = amdgpu.make_dma_base %A[%m_global_off_load, %k0], %a0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
                  %b0_base = amdgpu.make_dma_base %B[%n_global_off_load, %k0], %b0[%lds_sg_off, %c0] : !global_buffer_in, !tdm_tile -> !amdgpu.tdm_base<f16>
                  %a0_desc = amdgpu.make_dma_descriptor %a0_base
                    globalSize [%m_tile_size, %k0_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>
                  %b0_desc = amdgpu.make_dma_descriptor %b0_base
                    globalSize [%n_tile_size, %k0_tile_size]
                    globalStride [%KL, 1]
                    sharedSize [%c64, %c128]
                    padShared (%c4_i32 every %c64_i32)
                    : !amdgpu.tdm_base<f16> -> !amdgpu.tdm_descriptor<f16>

                  // Barriers here so we don't clobber group 0.
                  amdgpu.memory_counter_wait tensor(2) // 2 reads per group
                  gpu.barrier memfence [#gpu.address_space<workgroup>]

                  amdgpu.tensor_load_to_lds %a0_desc : !amdgpu.tdm_descriptor<f16>
                  amdgpu.tensor_load_to_lds %b0_desc : !amdgpu.tdm_descriptor<f16>

                  // Compute on matrix group 1.
                  %lhs1_slice = vector.transfer_read %a1_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                  %lhs1 = vector.shape_cast %lhs1_slice : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>
                  %rhs1_slice = vector.transfer_read %b1_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                  %rhs1 = vector.shape_cast %rhs1_slice : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>

                  // Note backwards lhs/rhs as above.
                  %acc1 = iree_codegen.inner_tiled ins(%rhs1, %lhs1) outs(%acc0)
                    {
                      indexing_maps = #contraction_accesses,
                      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
                      kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
                    } : vector<8x4x1x16xf16>, vector<8x4x1x16xf16> into vector<8x8x1x8xf32>
                  scf.yield %acc1 : vector<8x8x1x8xf32>
                }

                // Epilogue.
                amdgpu.memory_counter_wait tensor(0)
                gpu.barrier memfence [#gpu.address_space<workgroup>]

                // Compute on matrix group 0.
                %lhs_slice_epilogue = vector.transfer_read %a0_exp[%c0, %m_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                %lhs_epilogue = vector.shape_cast %lhs_slice_epilogue : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>
                %rhs_slice_epilogue = vector.transfer_read %b0_exp[%c0, %n_sg, %mn_lane, %c0, %k_lane, %c0], %poison_f16
                    {in_bounds = [true, true, true, true, true, true]}
                    : !matmul_in_tile, vector<8x1x1x4x1x16xf16>
                %rhs_epilogue = vector.shape_cast %rhs_slice_epilogue : vector<8x1x1x4x1x16xf16> to vector<8x4x1x16xf16>

                // Note backwards LHS/RHS as above
                %accFinal = iree_codegen.inner_tiled ins(%rhs_epilogue, %lhs_epilogue) outs(%accLoop)
                  {
                    indexing_maps = #contraction_accesses,
                    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
                    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
                  } : vector<8x4x1x16xf16>, vector<8x4x1x16xf16> into vector<8x8x1x8xf32>

                // Writeback
                %n_lane_store, %m_lane_store = affine.delinearize_index %laneid into (2, 16) : index, index
                // Don't want to do set up a complex transfer_read.
                // Note that Triton also uses a straightforward mask here ,though I
                // think we can optimize in an if statement for the happy path.
                // Note also that we've got M/N backwards because of the
                // tr(C) = tr(B) * tr(A) trick we pulled above.
                scf.for %n_outer = %c0 to %c8 step %c1 {
                  scf.for %m_outer = %c0 to %c8 step %c1 {
                    %m_offset_store = affine.linearize_index disjoint [%m_wg, %m_outer, %m_sg, %m_lane_store] by (8, 2, 16) : index
                    %m_valid = arith.cmpi ult, %m_offset_store, %M : index
                    %m_valid_bcast = vector.broadcast %m_valid : i1 to vector<8xi1>
                    %n_offset_store = affine.linearize_index disjoint [%n_wg, %n_outer, %n_sg, %n_lane_store, %c0] by (8, 2, 2, 8) : index
                    %n_elem = vector.step : vector<8xindex>
                    %n_offset_bcast = vector.broadcast %n_offset_store : index to vector<8xindex>
                    %N_bcast = vector.broadcast %N : index to vector<8xindex>
                    %proposed_n = arith.addi %n_offset_bcast, %n_elem : vector<8xindex>
                    %mask_n = arith.cmpi ult, %proposed_n, %N_bcast : vector<8xindex>
                    %mask = arith.andi %m_valid_bcast, %mask_n : vector<8xi1>

                    %data = vector.extract %accFinal[%n_outer, %m_outer, 0] : vector<8xf32> from vector<8x8x1x8xf32>
                    vector.maskedstore %C[%m_offset_store, %n_offset_store], %mask, %data : !global_buffer_out, vector<8xi1>, vector<8xf32>
                  } {unroll_loop}
                } {unroll_loop}
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
  util.func public @tdm_matmul_trb_wrap(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @tdm_matmul_trb_wrap(%input0: tensor<?x?xf16>, %input1: tensor<?x?xf16>) -> (%output0: tensor<?x?xf32>)"}} {
        %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f16 = hal.element_type<f16> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f16) encoding(%dense_row_major)
    %2 = arith.muli %0, %c2 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf16>{%0, %1} in !stream.resource<external>{%3}
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%5, %6]) type(%element_type_f16) encoding(%dense_row_major)
    %7 = arith.muli %5, %c2 : index
    %8 = arith.muli %7, %6 : index
    %9 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?x?xf16>{%5, %6} in !stream.resource<external>{%8}
    %10 = arith.muli %0, %c4 : index
    %11 = arith.muli %10, %5 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%11} => !stream.timepoint
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
    %28 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%4 as %arg2: !stream.resource<external>{%3}, %9 as %arg3: !stream.resource<external>{%8}, %result as %arg4: !stream.resource<external>{%11}) {
      stream.cmd.dispatch @tdm_matmul_trb::@rocm_hsaco_fb::@tdm_matmul_trb[%1, %6, %0, %5](%13, %15, %17, %19, %21, %23, %25, %27 : i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg2[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg3[%c0 for %8] : !stream.resource<external>{%8},
        wo %arg4[%c0 for %11] : !stream.resource<external>{%11}
      }
    } => !stream.timepoint
    %29 = stream.timepoint.await %28 => %result : !stream.resource<external>{%11}
    %30 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %29 : tensor<?x?xf32>{%0, %5} in !stream.resource<external>{%11} -> !hal.buffer_view
    util.return %30 : !hal.buffer_view
  }
}
