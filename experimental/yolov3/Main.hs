{-# LANGUAGE LambdaCase #-}

module Main where
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Spec
import Torch.Vision.Darknet.Forward
import Torch.NN
import Torch.Vision
import Torch.Tensor
import Torch.Functional
import Torch.DType
import Control.Exception.Safe

main = do
  mconfig <- readIniFile "test/yolov3.cfg"
  spec <- case mconfig of
    Right cfg@(DarknetConfig global layers) -> do
      case toDarknetSpec cfg of
        Right spec -> return spec
        Left err -> throwIO $ userError err
    Left err -> throwIO $ userError err
  net <- sample spec
  net' <- loadWeights net "test/yolov3.weights"
  readImageAsRGB8WithScaling "test/train.jpg" 416 416 True >>= \case
    Right input_data -> do
      let input_data' = divScalar (255::Float) (hwc2chw $ toType Float input_data)
      print $ nonMaxSuppression (snd $ forwardDarknet net' (Nothing, input_data')) 0.8 0.4
    Left err -> print err
