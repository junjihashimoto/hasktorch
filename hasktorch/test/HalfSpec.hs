{-# LANGUAGE ScopedTypeVariables #-}
module HalfSpec (spec) where

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import qualified Torch.DType as D
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Numeric.Half

spec :: Spec
spec = do
  it "add" $ do
    let x = [1,2,3,4] :: [Half]
        tx = asTensor x
        t = map asTensor x :: [Tensor]
        mysum = foldr (+) (asTensor (0::Half)) t
    dtype tx `shouldBe` D.Half
    shape tx `shouldBe` [4]
    asValue mysum `shouldBe` (10::Half)
    toInt mysum `shouldBe` 10

