# ......................................................................................
# MIT License

# Copyright (c) 2023 Pantelis I. Kaplanoglou

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

import numpy as np

# =========================================================================================================================
'''
    Standardizer that supports rank 3 and above tensors
'''
class CStandardScaler(object):
  # --------------------------------------------------------------------------------------
  def __init__(self):
    # ................................................................
    # // Fields \\    
    self.Mean = None
    self.Std = None
    # ................................................................    
  # --------------------------------------------------------------------------------------      
  def fit(self, p_nData, p_nAxisToKeep=-1):
    # Collect statistics with maximum precision
    p_nData = p_nData.astype(np.float64)
    nAxes = list(range(len(p_nData.shape)))
    if p_nAxisToKeep is None:
      nAxes = tuple(nAxes)
    else:
      if p_nAxisToKeep == -1:
        p_nAxisToKeep = nAxes[-1]
      
      nAxes.remove(p_nAxisToKeep)
      if len(nAxes) == 1:
        nAxes = nAxes[0]
      else:
        nAxes = tuple(nAxes)
    
    self.Mean = np.mean(p_nData, axis=nAxes)
    self.Std  = np.std(p_nData, axis=nAxes)
    print("  Standardization: mean/std shape:%s" % str(self.Mean.shape))
  # --------------------------------------------------------------------------------------    
  def fit_transform(self, p_nData, p_nAxisToKeep=-1):
    self.fit(p_nData, p_nAxisToKeep)
    return self.transform(p_nData)
  # --------------------------------------------------------------------------------------  
  def transform(self, p_nData):
    nStandardizedData = (p_nData - self.Mean) / self.Std
    return nStandardizedData.astype(p_nData.dtype)
  # --------------------------------------------------------------------------------------
  def inverse_transform(self, p_nData):
    nNonStandardizedData = (p_nData * self.Std) + self.Mean
    return nNonStandardizedData.astype(p_nData.dtype)
  # --------------------------------------------------------------------------------------
# =========================================================================================================================



if __name__ == "__main__":
  from sklearn.preprocessing import StandardScaler
  
  np.random.seed(2023)
  
  
  data = np.random.rand(100, 5)
  
  print(data.shape)
  
  #data = data.astype(np.float64)  # Same with sklearn
  data = data.astype(np.float32)  # sklearn is loosing precision at the 6th digit
  
  
  oScaler1 = StandardScaler()
  data1 = oScaler1.fit_transform(data)
  data1_inv = oScaler1.inverse_transform(data1)
  print(f"scaler mean/std shape:{oScaler1.mean_.shape}")
  print(np.sum(data), np.sum(data1_inv))
  
  oScaler2 = CStandardScaler()
  data2 = oScaler2.fit_transform(data)
  data2_inv = oScaler2.inverse_transform(data2)
  print(data1.dtype, data2.dtype)
  print(np.sum(data), np.sum(data2_inv))
  
  print("sklearn:%.8f" % np.sum(data1), "mllib:%.8f" % np.sum(data2))
  
  print("Rounding differences in mean")
  print(oScaler1.mean_ - oScaler2.Mean)
  print("Rounding differences in std")
  print(oScaler1.scale_ - oScaler2.Std)
  
    