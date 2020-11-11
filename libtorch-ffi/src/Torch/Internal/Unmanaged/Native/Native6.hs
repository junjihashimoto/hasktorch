
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Unmanaged.Native.Native6 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"


deg2rad__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
deg2rad__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::deg2rad_(
    *$(at::Tensor* _self)));
  }|]

deg2rad_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
deg2rad_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::deg2rad_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

scalar_tensor_so
  :: Ptr Scalar
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
scalar_tensor_so _s _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::scalar_tensor(
    *$(at::Scalar* _s)
  , *$(at::TensorOptions* _options)));
  }|]

scalar_tensor_s
  :: Ptr Scalar
  -> IO (Ptr Tensor)
scalar_tensor_s _s =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::scalar_tensor(
    *$(at::Scalar* _s)));
  }|]

rand_lNo
  :: Ptr IntArray
  -> Ptr DimnameList
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
rand_lNo _size _names _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<at::Dimname>* _names)
  , *$(at::TensorOptions* _options)));
  }|]

rand_lN
  :: Ptr IntArray
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
rand_lN _size _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<at::Dimname>* _names)));
  }|]

rand_lGNo
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr DimnameList
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
rand_lGNo _size _generator _names _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(std::vector<at::Dimname>* _names)
  , *$(at::TensorOptions* _options)));
  }|]

rand_lGN
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
rand_lGN _size _generator _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(std::vector<at::Dimname>* _names)));
  }|]

rand_lo
  :: Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
rand_lo _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

rand_l
  :: Ptr IntArray
  -> IO (Ptr Tensor)
rand_l _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)));
  }|]

rand_lGo
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
rand_lGo _size _generator _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(at::TensorOptions* _options)));
  }|]

rand_lG
  :: Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
rand_lG _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

rand_out_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
rand_out_tl _out _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand_out(
    *$(at::Tensor* _out)
  , *$(std::vector<int64_t>* _size)));
  }|]

rand_out_tlG
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
rand_out_tlG _out _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand_out(
    *$(at::Tensor* _out)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

rand_like_toM
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> MemoryFormat
  -> IO (Ptr Tensor)
rand_like_toM _self _options _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand_like(
    *$(at::Tensor* _self)
  , *$(at::TensorOptions* _options)
  , $(at::MemoryFormat _memory_format)));
  }|]

rand_like_to
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
rand_like_to _self _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand_like(
    *$(at::Tensor* _self)
  , *$(at::TensorOptions* _options)));
  }|]

rand_like_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
rand_like_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rand_like(
    *$(at::Tensor* _self)));
  }|]

randint_llo
  :: Int64
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_llo _high _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

randint_ll
  :: Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
randint_ll _high _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _high)
  , *$(std::vector<int64_t>* _size)));
  }|]

randint_llGo
  :: Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_llGo _high _size _generator _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(at::TensorOptions* _options)));
  }|]

randint_llG
  :: Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randint_llG _high _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randint_lllo
  :: Int64
  -> Int64
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_lllo _low _high _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

randint_lll
  :: Int64
  -> Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
randint_lll _low _high _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)));
  }|]

randint_lllGo
  :: Int64
  -> Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_lllGo _low _high _size _generator _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(at::TensorOptions* _options)));
  }|]

randint_lllG
  :: Int64
  -> Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randint_lllG _low _high _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint(
    $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randint_out_tll
  :: Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
randint_out_tll _out _high _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_out(
    *$(at::Tensor* _out)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)));
  }|]

randint_out_tllG
  :: Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randint_out_tllG _out _high _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_out(
    *$(at::Tensor* _out)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randint_out_tlll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
randint_out_tlll _out _low _high _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_out(
    *$(at::Tensor* _out)
  , $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)));
  }|]

randint_out_tlllG
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randint_out_tlllG _out _low _high _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_out(
    *$(at::Tensor* _out)
  , $(int64_t _low)
  , $(int64_t _high)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randint_like_tloM
  :: Ptr Tensor
  -> Int64
  -> Ptr TensorOptions
  -> MemoryFormat
  -> IO (Ptr Tensor)
randint_like_tloM _self _high _options _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _high)
  , *$(at::TensorOptions* _options)
  , $(at::MemoryFormat _memory_format)));
  }|]

randint_like_tlo
  :: Ptr Tensor
  -> Int64
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_like_tlo _self _high _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _high)
  , *$(at::TensorOptions* _options)));
  }|]

randint_like_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
randint_like_tl _self _high =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _high)));
  }|]

randint_like_tlloM
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr TensorOptions
  -> MemoryFormat
  -> IO (Ptr Tensor)
randint_like_tlloM _self _low _high _options _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _low)
  , $(int64_t _high)
  , *$(at::TensorOptions* _options)
  , $(at::MemoryFormat _memory_format)));
  }|]

randint_like_tllo
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randint_like_tllo _self _low _high _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _low)
  , $(int64_t _high)
  , *$(at::TensorOptions* _options)));
  }|]

randint_like_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
randint_like_tll _self _low _high =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randint_like(
    *$(at::Tensor* _self)
  , $(int64_t _low)
  , $(int64_t _high)));
  }|]

randn_lo
  :: Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randn_lo _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

randn_l
  :: Ptr IntArray
  -> IO (Ptr Tensor)
randn_l _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)));
  }|]

randn_lGo
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randn_lGo _size _generator _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(at::TensorOptions* _options)));
  }|]

randn_lG
  :: Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randn_lG _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randn_lNo
  :: Ptr IntArray
  -> Ptr DimnameList
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randn_lNo _size _names _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<at::Dimname>* _names)
  , *$(at::TensorOptions* _options)));
  }|]

randn_lN
  :: Ptr IntArray
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
randn_lN _size _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<at::Dimname>* _names)));
  }|]

randn_lGNo
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr DimnameList
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randn_lGNo _size _generator _names _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(std::vector<at::Dimname>* _names)
  , *$(at::TensorOptions* _options)));
  }|]

randn_lGN
  :: Ptr IntArray
  -> Ptr Generator
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
randn_lGN _size _generator _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn(
    *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)
  , *$(std::vector<at::Dimname>* _names)));
  }|]

randn_out_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
randn_out_tl _out _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn_out(
    *$(at::Tensor* _out)
  , *$(std::vector<int64_t>* _size)));
  }|]

randn_out_tlG
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr Generator
  -> IO (Ptr Tensor)
randn_out_tlG _out _size _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn_out(
    *$(at::Tensor* _out)
  , *$(std::vector<int64_t>* _size)
  , *$(at::Generator* _generator)));
  }|]

randn_like_toM
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> MemoryFormat
  -> IO (Ptr Tensor)
randn_like_toM _self _options _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn_like(
    *$(at::Tensor* _self)
  , *$(at::TensorOptions* _options)
  , $(at::MemoryFormat _memory_format)));
  }|]

randn_like_to
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randn_like_to _self _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn_like(
    *$(at::Tensor* _self)
  , *$(at::TensorOptions* _options)));
  }|]

randn_like_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
randn_like_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randn_like(
    *$(at::Tensor* _self)));
  }|]

randperm_lo
  :: Int64
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randperm_lo _n _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm(
    $(int64_t _n)
  , *$(at::TensorOptions* _options)));
  }|]

randperm_l
  :: Int64
  -> IO (Ptr Tensor)
randperm_l _n =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm(
    $(int64_t _n)));
  }|]

randperm_lGo
  :: Int64
  -> Ptr Generator
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
randperm_lGo _n _generator _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm(
    $(int64_t _n)
  , *$(at::Generator* _generator)
  , *$(at::TensorOptions* _options)));
  }|]

randperm_lG
  :: Int64
  -> Ptr Generator
  -> IO (Ptr Tensor)
randperm_lG _n _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm(
    $(int64_t _n)
  , *$(at::Generator* _generator)));
  }|]

randperm_out_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
randperm_out_tl _out _n =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm_out(
    *$(at::Tensor* _out)
  , $(int64_t _n)));
  }|]

randperm_out_tlG
  :: Ptr Tensor
  -> Int64
  -> Ptr Generator
  -> IO (Ptr Tensor)
randperm_out_tlG _out _n _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::randperm_out(
    *$(at::Tensor* _out)
  , $(int64_t _n)
  , *$(at::Generator* _generator)));
  }|]

range_ssso
  :: Ptr Scalar
  -> Ptr Scalar
  -> Ptr Scalar
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
range_ssso _start _end _step _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::range(
    *$(at::Scalar* _start)
  , *$(at::Scalar* _end)
  , *$(at::Scalar* _step)
  , *$(at::TensorOptions* _options)));
  }|]

range_sss
  :: Ptr Scalar
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
range_sss _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::range(
    *$(at::Scalar* _start)
  , *$(at::Scalar* _end)
  , *$(at::Scalar* _step)));
  }|]

range_out_tsss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
range_out_tsss _out _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::range_out(
    *$(at::Tensor* _out)
  , *$(at::Scalar* _start)
  , *$(at::Scalar* _end)
  , *$(at::Scalar* _step)));
  }|]

range_out_tss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
range_out_tss _out _start _end =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::range_out(
    *$(at::Tensor* _out)
  , *$(at::Scalar* _start)
  , *$(at::Scalar* _end)));
  }|]

reciprocal_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
reciprocal_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::reciprocal(
    *$(at::Tensor* _self)));
  }|]

reciprocal__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
reciprocal__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::reciprocal_(
    *$(at::Tensor* _self)));
  }|]

reciprocal_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
reciprocal_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::reciprocal_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

neg_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
neg_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::neg(
    *$(at::Tensor* _self)));
  }|]

neg__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
neg__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::neg_(
    *$(at::Tensor* _self)));
  }|]

neg_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
neg_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::neg_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

negative_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
negative_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::negative(
    *$(at::Tensor* _self)));
  }|]

negative__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
negative__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::negative_(
    *$(at::Tensor* _self)));
  }|]

negative_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
negative_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::negative_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

repeat_interleave_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
repeat_interleave_t _repeats =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::repeat_interleave(
    *$(at::Tensor* _repeats)));
  }|]

repeat_interleave_ttl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
repeat_interleave_ttl _self _repeats _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::repeat_interleave(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _repeats)
  , $(int64_t _dim)));
  }|]

repeat_interleave_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
repeat_interleave_tt _self _repeats =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::repeat_interleave(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _repeats)));
  }|]

repeat_interleave_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
repeat_interleave_tll _self _repeats _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::repeat_interleave(
    *$(at::Tensor* _self)
  , $(int64_t _repeats)
  , $(int64_t _dim)));
  }|]

repeat_interleave_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
repeat_interleave_tl _self _repeats =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::repeat_interleave(
    *$(at::Tensor* _self)
  , $(int64_t _repeats)));
  }|]

reshape_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
reshape_tl _self _shape =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::reshape(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _shape)));
  }|]

_mkldnn_reshape_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
_mkldnn_reshape_tl _self _shape =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_mkldnn_reshape(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _shape)));
  }|]

round_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
round_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::round(
    *$(at::Tensor* _self)));
  }|]

round__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
round__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::round_(
    *$(at::Tensor* _self)));
  }|]

round_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
round_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::round_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

rrelu_tssbG
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> CBool
  -> Ptr Generator
  -> IO (Ptr Tensor)
rrelu_tssbG _self _lower _upper _training _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)
  , $(bool _training)
  , *$(at::Generator* _generator)));
  }|]

rrelu_tssb
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> CBool
  -> IO (Ptr Tensor)
rrelu_tssb _self _lower _upper _training =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)
  , $(bool _training)));
  }|]

rrelu_tss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
rrelu_tss _self _lower _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)));
  }|]

rrelu_ts
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
rrelu_ts _self _lower =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)));
  }|]

rrelu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
rrelu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu(
    *$(at::Tensor* _self)));
  }|]

rrelu__tssbG
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> CBool
  -> Ptr Generator
  -> IO (Ptr Tensor)
rrelu__tssbG _self _lower _upper _training _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)
  , $(bool _training)
  , *$(at::Generator* _generator)));
  }|]

rrelu__tssb
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> CBool
  -> IO (Ptr Tensor)
rrelu__tssb _self _lower _upper _training =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)
  , $(bool _training)));
  }|]

rrelu__tss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
rrelu__tss _self _lower _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)
  , *$(at::Scalar* _upper)));
  }|]

rrelu__ts
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
rrelu__ts _self _lower =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lower)));
  }|]

rrelu__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
rrelu__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rrelu_(
    *$(at::Tensor* _self)));
  }|]

relu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
relu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::relu(
    *$(at::Tensor* _self)));
  }|]

relu__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
relu__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::relu_(
    *$(at::Tensor* _self)));
  }|]

prelu_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
prelu_tt _self _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prelu(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _weight)));
  }|]

prelu_backward_ttt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
prelu_backward_ttt _grad_output _self _weight =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::prelu_backward(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _self)
  , *$(at::Tensor* _weight)));
  }|]

gelu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
gelu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::gelu(
    *$(at::Tensor* _self)));
  }|]

gelu_backward_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
gelu_backward_tt _grad _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::gelu_backward(
    *$(at::Tensor* _grad)
  , *$(at::Tensor* _self)));
  }|]

infinitely_differentiable_gelu_backward_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
infinitely_differentiable_gelu_backward_tt _grad _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::infinitely_differentiable_gelu_backward(
    *$(at::Tensor* _grad)
  , *$(at::Tensor* _self)));
  }|]

hardshrink_ts
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
hardshrink_ts _self _lambd =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::hardshrink(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lambd)));
  }|]

hardshrink_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
hardshrink_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::hardshrink(
    *$(at::Tensor* _self)));
  }|]

hardshrink_backward_tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
hardshrink_backward_tts _grad_out _self _lambd =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::hardshrink_backward(
    *$(at::Tensor* _grad_out)
  , *$(at::Tensor* _self)
  , *$(at::Scalar* _lambd)));
  }|]

rsqrt_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
rsqrt_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rsqrt(
    *$(at::Tensor* _self)));
  }|]

rsqrt__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
rsqrt__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rsqrt_(
    *$(at::Tensor* _self)));
  }|]

rsqrt_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
rsqrt_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::rsqrt_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

select_tnl
  :: Ptr Tensor
  -> Ptr Dimname
  -> Int64
  -> IO (Ptr Tensor)
select_tnl _self _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::select(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(int64_t _index)));
  }|]

select_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
select_tll _self _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::select(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _index)));
  }|]

select_backward_tlll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
select_backward_tlll _grad _input_sizes _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::select_backward(
    *$(at::Tensor* _grad)
  , *$(std::vector<int64_t>* _input_sizes)
  , $(int64_t _dim)
  , $(int64_t _index)));
  }|]

selu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
selu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::selu(
    *$(at::Tensor* _self)));
  }|]

selu__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
selu__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::selu_(
    *$(at::Tensor* _self)));
  }|]

celu_ts
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
celu_ts _self _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::celu(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _alpha)));
  }|]

celu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
celu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::celu(
    *$(at::Tensor* _self)));
  }|]

celu__ts
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
celu__ts _self _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::celu_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _alpha)));
  }|]

celu__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
celu__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::celu_(
    *$(at::Tensor* _self)));
  }|]

silu_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
silu_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::silu(
    *$(at::Tensor* _self)));
  }|]

silu__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
silu__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::silu_(
    *$(at::Tensor* _self)));
  }|]

silu_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
silu_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::silu_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

silu_backward_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
silu_backward_tt _grad_output _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::silu_backward(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _self)));
  }|]

sigmoid_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sigmoid_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sigmoid(
    *$(at::Tensor* _self)));
  }|]

sigmoid__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sigmoid__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sigmoid_(
    *$(at::Tensor* _self)));
  }|]

sigmoid_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sigmoid_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sigmoid_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

logit_td
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
logit_td _self _eps =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit(
    *$(at::Tensor* _self)
  , $(double _eps)));
  }|]

logit_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
logit_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit(
    *$(at::Tensor* _self)));
  }|]

logit__td
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
logit__td _self _eps =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit_(
    *$(at::Tensor* _self)
  , $(double _eps)));
  }|]

logit__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
logit__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit_(
    *$(at::Tensor* _self)));
  }|]

logit_out_ttd
  :: Ptr Tensor
  -> Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
logit_out_ttd _out _self _eps =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , $(double _eps)));
  }|]

logit_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
logit_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::logit_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

sin_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sin_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin(
    *$(at::Tensor* _self)));
  }|]

sin__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sin__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin_(
    *$(at::Tensor* _self)));
  }|]

sin_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sin_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

sinh_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sinh_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh(
    *$(at::Tensor* _self)));
  }|]

sinh__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sinh__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh_(
    *$(at::Tensor* _self)));
  }|]

sinh_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sinh_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

detach_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
detach_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::detach(
    *$(at::Tensor* _self)));
  }|]

detach__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
detach__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::detach_(
    *$(at::Tensor* _self)));
  }|]

size_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
size_tl _self _dim =
  [C.throwBlock| int64_t { return (at::size(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

size_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
size_tn _self _dim =
  [C.throwBlock| int64_t { return (at::size(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

slice_tllll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tllll _self _dim _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)
  , $(int64_t _step)));
  }|]

slice_tlll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tlll _self _dim _start _end =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)));
  }|]

slice_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tll _self _dim _start =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)));
  }|]

slice_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
slice_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

slice_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
slice_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)));
  }|]

slice_backward_tlllll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_backward_tlllll _grad _input_sizes _dim _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice_backward(
    *$(at::Tensor* _grad)
  , *$(std::vector<int64_t>* _input_sizes)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)
  , $(int64_t _step)));
  }|]

slogdet_t
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
slogdet_t _self =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::slogdet(
    *$(at::Tensor* _self)));
  }|]

smm_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
smm_tt _self _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::smm(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _mat2)));
  }|]

softmax_tls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
softmax_tls _self _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

softmax_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
softmax_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

softmax_tns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
softmax_tns _self _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

softmax_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
softmax_tn _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

_softmax_tlb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
_softmax_tlb _self _dim _half_to_float =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _half_to_float)));
  }|]

_softmax_backward_data_ttlt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> IO (Ptr Tensor)
_softmax_backward_data_ttlt _grad_output _output _dim _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_softmax_backward_data(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _output)
  , $(int64_t _dim)
  , *$(at::Tensor* _self)));
  }|]

unsafe_split_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
unsafe_split_tll _self _split_size _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::unsafe_split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)
  , $(int64_t _dim)));
  }|]

unsafe_split_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr TensorList)
unsafe_split_tl _self _split_size =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::unsafe_split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)));
  }|]

split_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
split_tll _self _split_size _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)
  , $(int64_t _dim)));
  }|]

split_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr TensorList)
split_tl _self _split_size =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)));
  }|]

