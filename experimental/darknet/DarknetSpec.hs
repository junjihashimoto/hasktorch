{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ExtendedDefaultRules #-}

module Main(spec,main) where

import Test.Hspec
import Control.Exception.Safe
import Control.Monad.State.Strict

import Torch.Tensor
import Torch.TensorFactories
import Torch.NN
import Torch.Typed.NN (HasForward(..))
import GHC.Exts
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Spec
import Torch.Vision.Darknet.Forward
import qualified Torch.Functional.Internal as I
import GHC.Generics
import qualified System.IO
import qualified Data.ByteString.Lazy as B
import qualified Data.Map as M
import Torch.Functional

main = hspec spec

spec :: Spec
spec = do
  describe "index accesses" $ do
    it "index" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (v ! 2) :: Float
      r `shouldBe` 3.0
    it "index" $ do
      let v = asTensor (replicate 3 [1,2,3,4] :: [[Float]])
          r = asValue (v ! (Ellipsis, 0))
      r `shouldBe` [1.0::Float,1.0,1.0]
    it "index" $ do
      let v = asTensor (replicate 3 [1,2,3,4] :: [[Float]])
          r = asValue (v ! (Ellipsis, 0))
      r `shouldBe` [1.0::Float,1.0,1.0]
    it "indexPut" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (maskedFill v (1::Int) (5.0::Float))
      r `shouldBe` [1.0::Float,5.0,3.0,4.0]
    it "indexPut" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (maskedFill v (1::Int) (5.0::Float))
      r `shouldBe` [1.0::Float,5.0,3.0,4.0]
  describe "non max suppression" $ do
    it "xywh2xyxy" $ do
      let v = zeros' [1,504,4]
      shape (xywh2xyxy v) `shouldBe` [1,504,4]
    it "[...,:4]" $ do
      let v = zeros' [1,504,85]
      shape (v ! (Ellipsis, Slice (0,4))) `shouldBe` [1,504,4]
  describe "DarknetSpec" $ do
    it "Convolution" $ do
      let spec' = ConvolutionSpec {
                   input_filters = 3,
                   filters = 16,
                   layer_size = 3,
                   stride = 1,
                   activation = "leaky"
                 }
      layer <- sample spec'
      shape (forward layer (ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
    it "ConvolutionWithBatchNorm" $ do
      let spec' = ConvolutionWithBatchNormSpec {
                   input_filters = 3,
                   filters = 16,
                   layer_size = 3,
                   stride = 1,
                   activation = "leaky"
                 }
      layer <- sample spec'
      shape (forward layer (True,ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
      shape (forward layer (False,ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
    it "Read config" $ do
      mconfig <- readIniFile "test/yolov3-tiny.cfg"
      case mconfig of
        Right cfg@(DarknetConfig global layers) -> do
          length (toList layers) `shouldBe` 24
          case toDarknetSpec cfg of
            Right spec -> do
              length (show spec) > 0 `shouldBe` True
            Left err -> throwIO $ userError err
        Left err -> throwIO $ userError err
    it "Yolo:prediction" $ do
      let yolo = Torch.Vision.Darknet.Forward.Yolo [(23, 27), (37, 58), (81, 82)] 80 418
          pred = toPrediction yolo (ones' [1,255,10,10])
      shape (fromPrediction pred) `shouldBe` [1,3,10,10,85]
      shape (toX pred) `shouldBe` [1,3,10,10]
      shape (toPredClass pred) `shouldBe` [1,3,10,10,80]
    it "Yolo:grid" $ do
      shape (gridX 3) `shouldBe` [1,1,3,3]
      (asValue (gridX 3)::[[[[Float]]]]) `shouldBe` [[[[0.0,1.0,2.0],[0.0,1.0,2.0],[0.0,1.0,2.0]]]]
      (asValue (gridY 3)::[[[[Float]]]]) `shouldBe` [[[[0.0,0.0,0.0],[1.0,1.0,1.0],[2.0,2.0,2.0]]]]
      let scaled_anchors = toScaledAnchors [(81,82), (135,169)] (480.0/15.0)
      scaled_anchors `shouldBe` [(2.53125,2.5625),(4.21875,5.28125)]
      shape (toAnchorW scaled_anchors) `shouldBe` [1,2,1,1]
      (asValue (I.masked_select (asTensor ([1,2,3,4]::[Float]))  (asTensor ([True,False,True,False]::[Bool]))) :: [Float]) `shouldBe` [1.0,3.0]
      let v = zeros' [4]
      (asValue (I.index_put
               v
               ([asTensor ([2]::[Int])])
               (asTensor (12::Float))
               False)::[Float]) `shouldBe` [0.0,0.0,12.0,0.0]
      (asValue v ::[Float]) `shouldBe` [0.0,0.0,0.0,0.0]
    it "Read config" $ do
      mconfig <- readIniFile "test/yolov3.cfg"
      Right mconfig' <- readIniFile' "test/yolov3.cfg"
      outputChannels mconfig' 83 `shouldBe` Right 512
      spec <- case mconfig of
                Right cfg@(DarknetConfig global layers) -> do
                  length (toList layers) `shouldBe` 107
                  case toDarknetSpec cfg of
                    Right spec -> return spec
                    Left err -> throwIO $ userError err
                Left err -> throwIO $ userError err
      net <- sample spec
      net' <- loadWeights net "test/yolov3.weights"
      input_data <- System.IO.withFile "test/input_data" System.IO.ReadMode $ \h -> do
        load h (zeros' [1,3,416,416])
      shape (input_data) `shouldBe` [1,3,416,416]
      output_data0 <- System.IO.withFile "test/output_data0" System.IO.ReadMode $ \h -> do
        load h (zeros' [1,32,416,416])
      shape (output_data0) `shouldBe` [1,32,416,416]
      let output = fst (forwardDarknet' 107 net' (Nothing, input_data))
      asValue (mseLoss output_data0 (output M.! 0)) < (0.0001::Float) `shouldBe` True
      forM_ (Prelude.take 107 yolo_shapes) $ \(i,exp_shape) -> do
        let output = fst (forwardDarknet' i net' (Nothing, input_data))
            shape' = shape (output M.! (i-1))
        case exp_shape of
          [] -> (i, []) `shouldBe` (i, exp_shape)
          _ -> do
            (i, shape') `shouldBe` (i, exp_shape)
            output_data <- System.IO.withFile ("test/output_data" ++ show (i-1) ) System.IO.ReadMode $ \h -> do
              load h (zeros' exp_shape)
            let err_value =  asValue (mseLoss output_data (output M.! (i-1))) :: Float
            print i
            print err_value
            when (err_value > (0.0001::Float)) $ do
              print output_data
              print (output M.! (i-1))
            err_value < (0.0001::Float) `shouldBe` True
      let output' = snd (forwardDarknet net' (Nothing, input_data))
          amax = argmax (Dim 2) RemoveDim (output' ! (Slice (), Slice (), Slice (5,None)))
          conf = output' ! (Slice (), Slice (), 4)
      shape output' `shouldBe` [1,10647,85]
      shape amax `shouldBe` [1,10647]
      shape conf `shouldBe` [1,10647]

yolo_shapes = [
              (1, [1,32, 416, 416]),
              (2, [1,64, 208, 208]),
              (3, [1,32, 208, 208]),
              (4, [1,64, 208, 208]),
              (5, [1,64, 208, 208]),
              (6, [1,128, 104, 104]),
              (7, [1,64, 104, 104]),
              (8, [1,128, 104, 104]),
              (9, [1,128, 104, 104]),
              (10, [1, 64, 104, 104]),
              (11, [1, 128, 104, 104]),
              (12, [1, 128, 104, 104]),
              (13, [1, 256, 52, 52]),
              (14, [1, 128, 52, 52]),
              (15, [1, 256, 52, 52]),
              (16, [1, 256, 52, 52]),
              (17, [1, 128, 52, 52]),
              (18, [1, 256, 52, 52]),
              (19, [1, 256, 52, 52]),
              (20, [1, 128, 52, 52]),
              (21, [1, 256, 52, 52]),
              (22, [1, 256, 52, 52]),
              (23, [1, 128, 52, 52]),
              (24, [1, 256, 52, 52]),
              (25, [1, 256, 52, 52]),
              (26, [1, 128, 52, 52]),
              (27, [1, 256, 52, 52]),
              (28, [1, 256, 52, 52]),
              (29, [1, 128, 52, 52]),
              (30, [1, 256, 52, 52]),
              (31, [1, 256, 52, 52]),
              (32, [1, 128, 52, 52]),
              (33, [1, 256, 52, 52]),
              (34, [1, 256, 52, 52]),
              (35, [1, 128, 52, 52]),
              (36, [1, 256, 52, 52]),
              (37, [1, 256, 52, 52]),
              (38, [1, 512, 26, 26]),
              (39, [1, 256, 26, 26]),
              (40, [1, 512, 26, 26]),
              (41, [1, 512, 26, 26]),
              (42, [1, 256, 26, 26]),
              (43, [1, 512, 26, 26]),
              (44, [1, 512, 26, 26]),
              (45, [1, 256, 26, 26]),
              (46, [1, 512, 26, 26]),
              (47, [1, 512, 26, 26]),
              (48, [1, 256, 26, 26]),
              (49, [1, 512, 26, 26]),
              (50, [1, 512, 26, 26]),
              (51, [1, 256, 26, 26]),
              (52, [1, 512, 26, 26]),
              (53, [1, 512, 26, 26]),
              (54, [1, 256, 26, 26]),
              (55, [1, 512, 26, 26]),
              (56, [1, 512, 26, 26]),
              (57, [1, 256, 26, 26]),
              (58, [1, 512, 26, 26]),
              (59, [1, 512, 26, 26]),
              (60, [1, 256, 26, 26]),
              (61, [1, 512, 26, 26]),
              (62, [1, 512, 26, 26]),
              (63, [1, 1024, 13, 13]),
              (64, [1, 512, 13, 13]),
              (65, [1, 1024, 13, 13]),
              (66, [1, 1024, 13, 13]),
              (67, [1, 512, 13, 13]),
              (68, [1, 1024, 13, 13]),
              (69, [1, 1024, 13, 13]),
              (70, [1, 512, 13, 13]),
              (71, [1, 1024, 13, 13]),
              (72, [1, 1024, 13, 13]),
              (73, [1, 512, 13, 13]),
              (74, [1, 1024, 13, 13]),
              (75, [1, 1024, 13, 13]),
              (76, [1, 512, 13, 13]),
              (77, [1, 1024, 13, 13]),
              (78, [1, 512, 13, 13]),
              (79, [1, 1024, 13, 13]),
              (80, [1, 512, 13, 13]),
              (81, [1, 1024, 13, 13]),
              (82, [1, 255, 13, 13]),
              (83, [1, 507, 85]),
              (84, [1, 512, 13, 13]),
              (85, [1, 256, 13, 13]),
              (86, [1, 256, 26, 26]),
              (87, [1, 768, 26, 26]),
              (88, [1, 256, 26, 26]),
              (89, [1, 512, 26, 26]),
              (90, [1, 256, 26, 26]),
              (91, [1, 512, 26, 26]),
              (92, [1, 256, 26, 26]),
              (93, [1, 512, 26, 26]),
              (94, [1, 255, 26, 26]),
              (95, [1, 2028, 85]),
              (96, [1, 256, 26, 26]),
              (97, [1, 128, 26, 26]),
              (98, [1, 128, 52, 52]),
              (99, [1, 384, 52, 52]),
              (100, [1, 128, 52, 52]),
              (101, [1, 256, 52, 52]),
              (102, [1, 128, 52, 52]),
              (103, [1, 256, 52, 52]),
              (104, [1, 128, 52, 52]),
              (105, [1, 256, 52, 52]),
              (106, [1, 255, 52, 52]),
              (107, [1, 8112, 85])
              ]
