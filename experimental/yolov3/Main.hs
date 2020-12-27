{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}

module Main where

import qualified Codec.Picture as I
import Control.Monad (forM_, when, foldM)
import Control.Exception.Safe
import Torch hiding (conv2d, indexPut)
import Torch.Vision
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Forward
import Torch.Vision.Darknet.Spec
import System.Environment (getArgs)
import qualified Data.Map as M

labels :: [String]
labels = [
  "person",
  "bicycle",
  "car",
  "motorbike",
  "aeroplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "sofa",
  "pottedplant",
  "bed",
  "diningtable",
  "toilet",
  "tvmonitor",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
  ]

id2layer :: Int -> (Int,Int,Int,Int)
id2layer i =
  if i < layer0 then func 0 13 i
  else if i < layer0 + layer1 then func 1 26 (i-layer0)
  else func 2 52 (i-layer0-layer1)
  where
    func layer grid j =
      let gridy = (j `Prelude.mod` (grid*grid)) `Prelude.div` grid
          gridx = (j `Prelude.mod` (grid*grid)) `Prelude.mod` grid
          anchor = j `Prelude.div` (grid*grid)
      in (gridx,gridy,anchor,layer)
    layer0 = 13 * 13 * 3
    layer1 = 26 * 26 * 3
    layer2 = 52 * 52 * 3

main = do
  args <- getArgs
  when (length args /= 4) $ do
    putStrLn "Usage: yolov3 config-file weight-file input-image-file output-image-file"
  let config_file = args !! 0
      weight_file = args !! 1
      input_file = args !! 2
      output_file = args !! 3
      device = Device CUDA 0
      toDev = _toDevice device
      toHost = _toDevice (Device CPU 0)
    
  mconfig <- readIniFile config_file
  spec <- case mconfig of
    Right cfg@(DarknetConfig global layers) -> do
      case toDarknetSpec cfg of
        Right spec -> return spec
        Left err -> throwIO $ userError err
    Left err -> throwIO $ userError err
  net <- sample spec
  net' <- loadWeights net weight_file
  net'' <- updateDarknet net' $ \i layer -> do
    case layer of
      LConvolution c ->
        if i == 93 then do
          let c2 = (conv2d :: Convolution -> Conv2d) c
              w = conv2dWeight c2
              b = conv2dBias c2
              addr = [85*2+4,153,0,0] :: [Int]
              v = 0.74682 :: Float
              c' = c2 {conv2dWeight = IndependentTensor $ indexPut (toDependent w) addr v }
          print i
          print $ shape (toDependent w)
          return $ LConvolution c{conv2d =c'}
        else return layer
      _ -> return layer
--  let net''' = toDevice device net''
  
  readImageAsRGB8WithScaling input_file 416 416 True >>= \case
    Right (input_image, input_tensor) -> do
      let input_data' = divScalar (255 :: Float) (hwc2chw $ toType Float input_tensor)
          (outs,out) = forwardDarknet net'' (Nothing, input_data')
          outputs = nonMaxSuppression out 0.8 0.4
          func input =
            let (outs,_) = forwardDarknet net'' (Nothing, input)
            in (outs M.! 92) ! (0,153,11,13)
            --in (outs M.! 93) ! (0,85*2+4,11,13)
      print "--"
      print $ shape $ (outs M.! 92)
      print $ shape $ (outs M.! 93)
      print $ shape $ (outs M.! 92) ! (0,153,11,13)
      v <- smoothGrad 10 0.2 func input_data'
      let img = toType UInt8 $ chw2hwc $ clamp 0 255 $ (mulScalar (1024*100::Float) v) + (mulScalar (255::Float) input_data' )
      writePng "o2.png" $ img
      

      forM_ (zip [0..] outputs) $ \(i, output) -> do
        let [x0,y0,x1,y1,object_confidence,class_confidence,classid,ids] = map truncate (asValue output :: [Float])
        print $ (i, id2layer ids, map truncate (asValue output :: [Float]))
        drawString (show i ++ " " ++ labels !! classid) (x0+1) (y0+1) (255,255,255) (0,0,0) input_image
        drawRect x0 y0 x1 y1 (255,255,255) input_image
      let gridsize0 = 416 `Prelude.div` 13
          gridsize1 = 416 `Prelude.div` 26
          gridsize2 = 416 `Prelude.div` 52
          gridsize = 26
      forM_ [1..26] $ \i -> do
        drawLine (i * gridsize1) 0 (i * gridsize1) 416 (0,0,127) input_image
        drawLine 0 (i * gridsize1) 416 (i * gridsize1) (0,0,127) input_image
      drawRect (11 * gridsize1) (13 * gridsize1) (12 * gridsize1) (14 * gridsize1) (255,0,0) input_image
      I.writePng output_file input_image
    Left err -> print err
