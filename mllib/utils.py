# -*- coding: utf-8 -*-
# ......................................................................................
# MIT License

# Copyright (c) 2020-2023 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................

IS_TENSORFLOW = False

import os
import random
import numpy as np


if IS_TENSORFLOW:
  import tensorflow as tf  



# --------------------------------------------------------------------------------------
# We are seeding the number generators to get some amount of determinism for the whole ML training process. 
# This is not ensuring 100% deterministic reproduction of an experiment in GPUs
def RandomSeed(p_nSeed=2022):
  random.seed(p_nSeed)
  os.environ['PYTHONHASHSEED'] = str(p_nSeed)
  np.random.seed(p_nSeed)   
  if IS_TENSORFLOW: 
    tf.compat.v1.reset_default_graph()
    #tf.compat.v1.set_random_seed(cls.SEED)
    tf.random.set_seed(p_nSeed)
    tf.keras.utils.set_random_seed(p_nSeed)
    
  print("Random seed set to %d" % p_nSeed)
# --------------------------------------------------------------------------------------
'''
  Checks if the p_sSettingsName is inside the settings dictionary p_dConfig
  and returns its value, otherwise the p_oDefault value
'''
def DefaultSetting(p_dConfig, p_sSettingName, p_oDefault=None):
  if p_sSettingName in p_dConfig:
    return p_dConfig[p_sSettingName]
  else:
    return p_oDefault
# --------------------------------------------------------------------------------------    


# -----------------------------------------------------------------------------
def PrintTensor(p_sTitle, p_nTensor, p_sFormat="%.3f"):
  # ................................................
  def printElement(p_nElement, p_bIsScalar):
    if p_bIsScalar:
      print(p_sFormat % p_nElement, end=" ")
    else:
      print(np.array2string(p_nElement, separator=",", formatter={'float': lambda x: p_sFormat % x}), end=" ")
  # ................................................
  def strBoxLeft(p_nIndex, p_nCount):
    if (p_nIndex == 0):
      return "┌ "
    elif (p_nIndex == (p_nCount-1)):
      return "└ "
    else:
      return "│ "
  # ................................................
  def strBoxRight(p_nIndex, p_nCount):
    if (p_nIndex == 0):
      return "┐ "
    elif (p_nIndex == (p_nCount-1)):
      return "┘ "
    else:
      return "│ "
  # ................................................
  
  
  if len(p_nTensor.shape) == 3:
    p_nTensor = p_nTensor[np.newaxis, ...]
  elif len(p_nTensor.shape) == 2:
    p_nTensor = p_nTensor[np.newaxis, ..., np.newaxis]
  elif len(p_nTensor.shape) == 5:
    p_nTensor = p_nTensor[..., np.newaxis]
      
  nSampleIndex = 0
  nCount, nGridRows, nGridCols = p_nTensor.shape[0:3]
  nSliceCoordDigits = len(str(nGridRows))
  if len(str(nGridCols)) > nSliceCoordDigits:
    nSliceCoordDigits = len(str(nGridCols))
        
  bIsGridOfTensors = (len(p_nTensor.shape) == 6)
  if bIsGridOfTensors:
    nWindowRows, nWindowCols = p_nTensor.shape[3:5]

  
  bIsScalar = (p_nTensor.shape[-1] == 1) 
  
  nSpaces = nSliceCoordDigits*2 + 5
  sSliceHeaders = [ "X" + " "*(nSpaces-3) + "= ", 
                    " %" + str(nSliceCoordDigits) + "d" ",%" + str(nSliceCoordDigits) + "d   ",
                    " "*nSpaces]
    
  print("-"*70)
  print("%s shape:%s" % (p_sTitle, p_nTensor.shape))
  while nSampleIndex < nCount:
    print("  Sample:#%d" % nSampleIndex)
    for nRow in range(0, nGridRows):
      if bIsGridOfTensors:
        for nY in range(nWindowRows):
          for nCol in range(0, nGridCols):
            if (nY == 0):
              print(sSliceHeaders[0] + strBoxLeft(nY, nWindowRows), end="")
            elif (nY == 1):
              sBaseStr = sSliceHeaders[1]  % (nRow, nCol) 
              print(sBaseStr + strBoxLeft(nY, nWindowRows), end="")
            else:
              print(sSliceHeaders[2] + strBoxLeft(nY, nWindowRows) , end="")
            for nX in range(nWindowCols):
              printElement(p_nTensor[nSampleIndex, nRow, nCol, nY, nX, ...], bIsScalar)
            
            print(strBoxRight(nY, nWindowRows), end="")
          print("")
        print("")
      else:
        print(strBoxLeft(nRow, nGridRows), end="")
        for nCol in range(0, nGridCols):
          printElement(p_nTensor[nSampleIndex, nRow, nCol, ...], bIsScalar)
        print(strBoxRight(nRow, nGridRows))
    print("."*60)
    nSampleIndex += 1
# -----------------------------------------------------------------------------



if __name__ == "__main__":
  SAMPLE_COUNT      = 4
  GRID_SIZE         = 3
  FEATURE_COUNT     = 1
  RANK              = 6
  
  RandomSeed(2023)
  if RANK == 6:
    nInput = np.random.rand(SAMPLE_COUNT,GRID_SIZE,GRID_SIZE, 3, 3, FEATURE_COUNT).astype(np.float32)
  elif RANK == 5:
    nInput = np.random.rand(SAMPLE_COUNT,GRID_SIZE,GRID_SIZE, 2, 2).astype(np.float32)
  elif RANK == 4:
    nInput = np.random.rand(SAMPLE_COUNT,GRID_SIZE,GRID_SIZE, FEATURE_COUNT).astype(np.float32)
  elif RANK == 3:
    nInput = np.random.rand(GRID_SIZE, GRID_SIZE, FEATURE_COUNT).astype(np.float32)
  elif RANK == 2:
    nInput = np.random.rand(GRID_SIZE,GRID_SIZE).astype(np.float32)
    
  PrintTensor("Data", nInput)
    

