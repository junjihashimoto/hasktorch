{-# LANGUAGE LambdaCase #-}

module Main where

import Torch.GraduallyTyped

main = do
  testMHA >>= print
  
  
