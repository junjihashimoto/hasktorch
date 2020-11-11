
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Managed.Native.Native9 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Native.Native9 as Unmanaged


addmm_ttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
addmm_ttts = cast4 Unmanaged.addmm_ttts

addmm_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
addmm_ttt = cast3 Unmanaged.addmm_ttt

sparse_coo_tensor_lo
  :: ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
sparse_coo_tensor_lo = cast2 Unmanaged.sparse_coo_tensor_lo

sparse_coo_tensor_tto
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
sparse_coo_tensor_tto = cast3 Unmanaged.sparse_coo_tensor_tto

sparse_coo_tensor_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
sparse_coo_tensor_tt = cast2 Unmanaged.sparse_coo_tensor_tt

sparse_coo_tensor_ttlo
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
sparse_coo_tensor_ttlo = cast4 Unmanaged.sparse_coo_tensor_ttlo

sparse_coo_tensor_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
sparse_coo_tensor_ttl = cast3 Unmanaged.sparse_coo_tensor_ttl

_sparse_coo_tensor_unsafe_ttlo
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
_sparse_coo_tensor_unsafe_ttlo = cast4 Unmanaged._sparse_coo_tensor_unsafe_ttlo

_sparse_coo_tensor_unsafe_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
_sparse_coo_tensor_unsafe_ttl = cast3 Unmanaged._sparse_coo_tensor_unsafe_ttl

_validate_sparse_coo_tensor_args_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (())
_validate_sparse_coo_tensor_args_ttl = cast3 Unmanaged._validate_sparse_coo_tensor_args_ttl

_sparse_coo_tensor_with_dims_lllo
  :: Int64
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
_sparse_coo_tensor_with_dims_lllo = cast4 Unmanaged._sparse_coo_tensor_with_dims_lllo

_sparse_coo_tensor_with_dims_and_tensors_llltto
  :: Int64
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
_sparse_coo_tensor_with_dims_and_tensors_llltto = cast6 Unmanaged._sparse_coo_tensor_with_dims_and_tensors_llltto

to_dense_backward_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
to_dense_backward_tt = cast2 Unmanaged.to_dense_backward_tt

hspmm_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
hspmm_out_ttt = cast3 Unmanaged.hspmm_out_ttt

hspmm_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
hspmm_tt = cast2 Unmanaged.hspmm_tt

copy_sparse_to_sparse__ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
copy_sparse_to_sparse__ttb = cast3 Unmanaged.copy_sparse_to_sparse__ttb

copy_sparse_to_sparse__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
copy_sparse_to_sparse__tt = cast2 Unmanaged.copy_sparse_to_sparse__tt

unbind_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr TensorList)
unbind_tl = cast2 Unmanaged.unbind_tl

unbind_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr TensorList)
unbind_t = cast1 Unmanaged.unbind_t

unbind_tn
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr TensorList)
unbind_tn = cast2 Unmanaged.unbind_tn

mkldnn_reorder_conv2d_weight_tllll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv2d_weight_tllll = cast5 Unmanaged.mkldnn_reorder_conv2d_weight_tllll

mkldnn_reorder_conv2d_weight_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv2d_weight_tlll = cast4 Unmanaged.mkldnn_reorder_conv2d_weight_tlll

mkldnn_reorder_conv2d_weight_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv2d_weight_tll = cast3 Unmanaged.mkldnn_reorder_conv2d_weight_tll

mkldnn_reorder_conv2d_weight_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv2d_weight_tl = cast2 Unmanaged.mkldnn_reorder_conv2d_weight_tl

mkldnn_reorder_conv2d_weight_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv2d_weight_t = cast1 Unmanaged.mkldnn_reorder_conv2d_weight_t

mkldnn_reorder_conv3d_weight_tllll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv3d_weight_tllll = cast5 Unmanaged.mkldnn_reorder_conv3d_weight_tllll

mkldnn_reorder_conv3d_weight_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv3d_weight_tlll = cast4 Unmanaged.mkldnn_reorder_conv3d_weight_tlll

mkldnn_reorder_conv3d_weight_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv3d_weight_tll = cast3 Unmanaged.mkldnn_reorder_conv3d_weight_tll

mkldnn_reorder_conv3d_weight_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv3d_weight_tl = cast2 Unmanaged.mkldnn_reorder_conv3d_weight_tl

mkldnn_reorder_conv3d_weight_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
mkldnn_reorder_conv3d_weight_t = cast1 Unmanaged.mkldnn_reorder_conv3d_weight_t

to_mkldnn_backward_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
to_mkldnn_backward_tt = cast2 Unmanaged.to_mkldnn_backward_tt

quantize_per_tensor_tdls
  :: ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
quantize_per_tensor_tdls = cast4 Unmanaged.quantize_per_tensor_tdls

quantize_per_tensor_ltts
  :: ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr TensorList)
quantize_per_tensor_ltts = cast4 Unmanaged.quantize_per_tensor_ltts

quantize_per_channel_tttls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
quantize_per_channel_tttls = cast5 Unmanaged.quantize_per_channel_tttls

dequantize_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
dequantize_t = cast1 Unmanaged.dequantize_t

dequantize_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr TensorList)
dequantize_l = cast1 Unmanaged.dequantize_l

q_scale_t
  :: ForeignPtr Tensor
  -> IO (CDouble)
q_scale_t = cast1 Unmanaged.q_scale_t

q_zero_point_t
  :: ForeignPtr Tensor
  -> IO (Int64)
q_zero_point_t = cast1 Unmanaged.q_zero_point_t

q_per_channel_scales_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
q_per_channel_scales_t = cast1 Unmanaged.q_per_channel_scales_t

q_per_channel_zero_points_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
q_per_channel_zero_points_t = cast1 Unmanaged.q_per_channel_zero_points_t

q_per_channel_axis_t
  :: ForeignPtr Tensor
  -> IO (Int64)
q_per_channel_axis_t = cast1 Unmanaged.q_per_channel_axis_t

int_repr_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
int_repr_t = cast1 Unmanaged.int_repr_t

_make_per_tensor_quantized_tensor_tdl
  :: ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> IO (ForeignPtr Tensor)
_make_per_tensor_quantized_tensor_tdl = cast3 Unmanaged._make_per_tensor_quantized_tensor_tdl

_make_per_channel_quantized_tensor_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
_make_per_channel_quantized_tensor_tttl = cast4 Unmanaged._make_per_channel_quantized_tensor_tttl

fake_quantize_per_tensor_affine_tdlll
  :: ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fake_quantize_per_tensor_affine_tdlll = cast5 Unmanaged.fake_quantize_per_tensor_affine_tdlll

fake_quantize_per_tensor_affine_backward_ttdlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fake_quantize_per_tensor_affine_backward_ttdlll = cast6 Unmanaged.fake_quantize_per_tensor_affine_backward_ttdlll

_fake_quantize_learnable_per_tensor_affine_tttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
_fake_quantize_learnable_per_tensor_affine_tttll = cast5 Unmanaged._fake_quantize_learnable_per_tensor_affine_tttll

_fake_quantize_learnable_per_tensor_affine_backward_ttttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_fake_quantize_learnable_per_tensor_affine_backward_ttttll = cast6 Unmanaged._fake_quantize_learnable_per_tensor_affine_backward_ttttll

fake_quantize_per_channel_affine_tttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fake_quantize_per_channel_affine_tttlll = cast6 Unmanaged.fake_quantize_per_channel_affine_tttlll

fake_quantize_per_channel_affine_backward_ttttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fake_quantize_per_channel_affine_backward_ttttlll = cast7 Unmanaged.fake_quantize_per_channel_affine_backward_ttttlll

_fake_quantize_learnable_per_channel_affine_tttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
_fake_quantize_learnable_per_channel_affine_tttlll = cast6 Unmanaged._fake_quantize_learnable_per_channel_affine_tttlll

_fake_quantize_learnable_per_channel_affine_backward_ttttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_fake_quantize_learnable_per_channel_affine_backward_ttttlll = cast7 Unmanaged._fake_quantize_learnable_per_channel_affine_backward_ttttlll

_choose_qparams_per_tensor_tb
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(CDouble,Int64)))
_choose_qparams_per_tensor_tb = cast2 Unmanaged._choose_qparams_per_tensor_tb

_choose_qparams_per_tensor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(CDouble,Int64)))
_choose_qparams_per_tensor_t = cast1 Unmanaged._choose_qparams_per_tensor_t

_saturate_weight_to_fp16_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_saturate_weight_to_fp16_t = cast1 Unmanaged._saturate_weight_to_fp16_t

choose_qparams_optimized_tlldl
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CDouble
  -> Int64
  -> IO (ForeignPtr (StdTuple '(CDouble,CDouble)))
choose_qparams_optimized_tlldl = cast5 Unmanaged.choose_qparams_optimized_tlldl

meshgrid_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr TensorList)
meshgrid_l = cast1 Unmanaged.meshgrid_l

cartesian_prod_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr Tensor)
cartesian_prod_l = cast1 Unmanaged.cartesian_prod_l

combinations_tlb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
combinations_tlb = cast3 Unmanaged.combinations_tlb

combinations_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
combinations_tl = cast2 Unmanaged.combinations_tl

combinations_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
combinations_t = cast1 Unmanaged.combinations_t

result_type_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ScalarType)
result_type_tt = cast2 Unmanaged.result_type_tt

result_type_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ScalarType)
result_type_ts = cast2 Unmanaged.result_type_ts

result_type_st
  :: ForeignPtr Scalar
  -> ForeignPtr Tensor
  -> IO (ScalarType)
result_type_st = cast2 Unmanaged.result_type_st

result_type_ss
  :: ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ScalarType)
result_type_ss = cast2 Unmanaged.result_type_ss

can_cast_ss
  :: ScalarType
  -> ScalarType
  -> IO (CBool)
can_cast_ss = cast2 Unmanaged.can_cast_ss

promote_types_ss
  :: ScalarType
  -> ScalarType
  -> IO (ScalarType)
promote_types_ss = cast2 Unmanaged.promote_types_ss

_local_scalar_dense_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Scalar)
_local_scalar_dense_t = cast1 Unmanaged._local_scalar_dense_t

_thnn_fused_lstm_cell_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_thnn_fused_lstm_cell_ttttt = cast5 Unmanaged._thnn_fused_lstm_cell_ttttt

_thnn_fused_lstm_cell_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_thnn_fused_lstm_cell_tttt = cast4 Unmanaged._thnn_fused_lstm_cell_tttt

_thnn_fused_lstm_cell_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_thnn_fused_lstm_cell_ttt = cast3 Unmanaged._thnn_fused_lstm_cell_ttt

_thnn_fused_lstm_cell_backward_tttttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)))
_thnn_fused_lstm_cell_backward_tttttb = cast6 Unmanaged._thnn_fused_lstm_cell_backward_tttttb

_thnn_differentiable_lstm_cell_backward_tttttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)))
_thnn_differentiable_lstm_cell_backward_tttttttt = cast8 Unmanaged._thnn_differentiable_lstm_cell_backward_tttttttt

_thnn_fused_gru_cell_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_thnn_fused_gru_cell_ttttt = cast5 Unmanaged._thnn_fused_gru_cell_ttttt

_thnn_fused_gru_cell_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_thnn_fused_gru_cell_tttt = cast4 Unmanaged._thnn_fused_gru_cell_tttt

_thnn_fused_gru_cell_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_thnn_fused_gru_cell_ttt = cast3 Unmanaged._thnn_fused_gru_cell_ttt

_thnn_fused_gru_cell_backward_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)))
_thnn_fused_gru_cell_backward_ttb = cast3 Unmanaged._thnn_fused_gru_cell_backward_ttb

_thnn_differentiable_gru_cell_backward_tttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)))
_thnn_differentiable_gru_cell_backward_tttttt = cast6 Unmanaged._thnn_differentiable_gru_cell_backward_tttttt

lstm_tllbldbbb
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
lstm_tllbldbbb = cast9 Unmanaged.lstm_tllbldbbb

lstm_ttllbldbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
lstm_ttllbldbb = cast9 Unmanaged.lstm_ttllbldbb

gru_ttlbldbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
gru_ttlbldbbb = cast9 Unmanaged.gru_ttlbldbbb

gru_tttlbldbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
gru_tttlbldbb = cast9 Unmanaged.gru_tttlbldbb

rnn_tanh_ttlbldbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
rnn_tanh_ttlbldbbb = cast9 Unmanaged.rnn_tanh_ttlbldbbb

rnn_tanh_tttlbldbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
rnn_tanh_tttlbldbb = cast9 Unmanaged.rnn_tanh_tttlbldbb

rnn_relu_ttlbldbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
rnn_relu_ttlbldbbb = cast9 Unmanaged.rnn_relu_ttlbldbbb

rnn_relu_tttlbldbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> CBool
  -> Int64
  -> CDouble
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
rnn_relu_tttlbldbb = cast9 Unmanaged.rnn_relu_tttlbldbb

lstm_cell_tltttt
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
lstm_cell_tltttt = cast6 Unmanaged.lstm_cell_tltttt

lstm_cell_tlttt
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
lstm_cell_tlttt = cast5 Unmanaged.lstm_cell_tlttt

lstm_cell_tltt
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
lstm_cell_tltt = cast4 Unmanaged.lstm_cell_tltt

gru_cell_tttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gru_cell_tttttt = cast6 Unmanaged.gru_cell_tttttt

gru_cell_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gru_cell_ttttt = cast5 Unmanaged.gru_cell_ttttt

gru_cell_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
gru_cell_tttt = cast4 Unmanaged.gru_cell_tttt

rnn_tanh_cell_tttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_tanh_cell_tttttt = cast6 Unmanaged.rnn_tanh_cell_tttttt

rnn_tanh_cell_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_tanh_cell_ttttt = cast5 Unmanaged.rnn_tanh_cell_ttttt

rnn_tanh_cell_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_tanh_cell_tttt = cast4 Unmanaged.rnn_tanh_cell_tttt

rnn_relu_cell_tttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_relu_cell_tttttt = cast6 Unmanaged.rnn_relu_cell_tttttt

rnn_relu_cell_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_relu_cell_ttttt = cast5 Unmanaged.rnn_relu_cell_ttttt

rnn_relu_cell_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
rnn_relu_cell_tttt = cast4 Unmanaged.rnn_relu_cell_tttt

quantized_lstm_cell_tlttttttttssss
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
quantized_lstm_cell_tlttttttttssss = cast14 Unmanaged.quantized_lstm_cell_tlttttttttssss

quantized_gru_cell_ttttttttttssss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
quantized_gru_cell_ttttttttttssss = cast14 Unmanaged.quantized_gru_cell_ttttttttttssss

quantized_rnn_relu_cell_ttttttttttssss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
quantized_rnn_relu_cell_ttttttttttssss = cast14 Unmanaged.quantized_rnn_relu_cell_ttttttttttssss

quantized_rnn_tanh_cell_ttttttttttssss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
quantized_rnn_tanh_cell_ttttttttttssss = cast14 Unmanaged.quantized_rnn_tanh_cell_ttttttttttssss

_pack_padded_sequence_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_pack_padded_sequence_ttb = cast3 Unmanaged._pack_padded_sequence_ttb

_pack_padded_sequence_backward_tltb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
_pack_padded_sequence_backward_tltb = cast4 Unmanaged._pack_padded_sequence_backward_tltb

_pad_packed_sequence_ttbsl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> ForeignPtr Scalar
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
_pad_packed_sequence_ttbsl = cast5 Unmanaged._pad_packed_sequence_ttbsl

masked_fill_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
masked_fill_tts = cast3 Unmanaged.masked_fill_tts

masked_fill_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
masked_fill_ttt = cast3 Unmanaged.masked_fill_ttt

masked_scatter_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
masked_scatter_ttt = cast3 Unmanaged.masked_scatter_ttt

index_add_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_add_tltt = cast4 Unmanaged.index_add_tltt

index_add_tntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_add_tntt = cast4 Unmanaged.index_add_tntt

index_fill_tlts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
index_fill_tlts = cast4 Unmanaged.index_fill_tlts

index_fill_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_fill_tltt = cast4 Unmanaged.index_fill_tltt

index_fill_tnts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
index_fill_tnts = cast4 Unmanaged.index_fill_tnts

index_fill_tntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
index_fill_tntt = cast4 Unmanaged.index_fill_tntt

scatter_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
scatter_tltt = cast4 Unmanaged.scatter_tltt

scatter_tlts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
scatter_tlts = cast4 Unmanaged.scatter_tlts

scatter_tntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
scatter_tntt = cast4 Unmanaged.scatter_tntt

scatter_tnts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
scatter_tnts = cast4 Unmanaged.scatter_tnts

scatter_add_tltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
scatter_add_tltt = cast4 Unmanaged.scatter_add_tltt

scatter_add_tntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
scatter_add_tntt = cast4 Unmanaged.scatter_add_tntt

bitwise_and_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_and_out_ttt = cast3 Unmanaged.bitwise_and_out_ttt

bitwise_and_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_and_out_tts = cast3 Unmanaged.bitwise_and_out_tts

bitwise_and_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_and_ts = cast2 Unmanaged.bitwise_and_ts

bitwise_and_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_and_tt = cast2 Unmanaged.bitwise_and_tt

__and___ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
__and___ts = cast2 Unmanaged.__and___ts

__and___tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
__and___tt = cast2 Unmanaged.__and___tt

bitwise_or_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_or_out_ttt = cast3 Unmanaged.bitwise_or_out_ttt

bitwise_or_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_or_out_tts = cast3 Unmanaged.bitwise_or_out_tts

bitwise_or_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_or_ts = cast2 Unmanaged.bitwise_or_ts

bitwise_or_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_or_tt = cast2 Unmanaged.bitwise_or_tt

__or___ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
__or___ts = cast2 Unmanaged.__or___ts

__or___tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
__or___tt = cast2 Unmanaged.__or___tt

bitwise_xor_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_xor_out_ttt = cast3 Unmanaged.bitwise_xor_out_ttt

bitwise_xor_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_xor_out_tts = cast3 Unmanaged.bitwise_xor_out_tts

bitwise_xor_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
bitwise_xor_ts = cast2 Unmanaged.bitwise_xor_ts

bitwise_xor_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_xor_tt = cast2 Unmanaged.bitwise_xor_tt

__xor___ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
__xor___ts = cast2 Unmanaged.__xor___ts

__xor___tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
__xor___tt = cast2 Unmanaged.__xor___tt

__lshift___ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
__lshift___ts = cast2 Unmanaged.__lshift___ts

__lshift___tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
__lshift___tt = cast2 Unmanaged.__lshift___tt

__rshift___ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
__rshift___ts = cast2 Unmanaged.__rshift___ts

__rshift___tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
__rshift___tt = cast2 Unmanaged.__rshift___tt

addbmm_out_ttttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
addbmm_out_ttttss = cast6 Unmanaged.addbmm_out_ttttss

addbmm_out_tttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
addbmm_out_tttts = cast5 Unmanaged.addbmm_out_tttts

addbmm_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
addbmm_out_tttt = cast4 Unmanaged.addbmm_out_tttt

addbmm_tttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
addbmm_tttss = cast5 Unmanaged.addbmm_tttss

addbmm_ttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
addbmm_ttts = cast4 Unmanaged.addbmm_ttts

addbmm_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
addbmm_ttt = cast3 Unmanaged.addbmm_ttt

diag_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
diag_out_ttl = cast3 Unmanaged.diag_out_ttl

diag_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
diag_out_tt = cast2 Unmanaged.diag_out_tt

diag_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
diag_tl = cast2 Unmanaged.diag_tl

diag_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
diag_t = cast1 Unmanaged.diag_t

diag_backward_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
diag_backward_tll = cast3 Unmanaged.diag_backward_tll

