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
import mllib as ml
from sklearn.model_selection import train_test_split    # import a standalone procedure function from the pacakge


# =========================================================================================================================
class CDataSetBase(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_sName=None, p_nRandomSeed=None, p_bIsClassification=True):
    # ................................................................
    # // Fields \\
    self.Name               = p_sName
    self.RandomSeed         = p_nRandomSeed
    self.IsClassification   = p_bIsClassification
    
    self.FeatureCount       = None
    self.ClassCount         = None
    self.SampleCount        = None
    
    self.Samples            = None
    self.Targets            = None
    
    self.TSSamples          = None
    self.TSTargets          = None
    self.TSSampleCount      = 0
    
    self.VSSamples          = None
    self.VSTargets          = None
    self.VSSampleCount      = 0
    
    self.UTSamples          = None
    self.UTTargets          = None
    self.UTSampleCount      = None
    
    # ................................................................
    if self.RandomSeed is not None:
        ml.RandomSeed(self.RandomSeed)
  # --------------------------------------------------------------------------------------
  @property
  def Labels(self):
    return self.Targets
  # --------------------------------------------------------------------------------------
  @property
  def TSLabels(self):
    return self.TSTargets
  # --------------------------------------------------------------------------------------
  @property
  def VSLabels(self):
    return self.VSTargets
  # --------------------------------------------------------------------------------------
  @property
  def UTLabels(self):
    return self.UTTargets
  # --------------------------------------------------------------------------------------
  def TrainingSet(self, p_nSamples, p_nTargets):
    self.TSSamples = p_nSamples
    self.TSTargets = p_nTargets
    self.countSamples()
    
    # Feature count is calculated on samples that are flattened as vectors
    if self.FeatureCount is None:
      self.FeatureCount = np.prod(self.TSSamples.shape[1:])
      
    if self.ClassCount is None:
      if self.IsClassification:        
        self.ClassCount = len(np.unique(self.TSTargets))
      else:
        self.ClassCount = 0
  # --------------------------------------------------------------------------------------
  def ValidationSet(self, p_nSamples, p_nTargets):
    self.VSSamples = p_nSamples
    self.VSTargets = p_nTargets
    self.countSamples()        
  # --------------------------------------------------------------------------------------
  def UnknownTestSet(self, p_nSamples, p_nTargets):
    self.UTSamples = p_nSamples
    self.UTTargets = p_nTargets
    self.countSamples()           
  # --------------------------------------------------------------------------------------
  def PrintInfo(self):
    print("Dataset [%s]" % self.Name)
    print("  |__ FeatureCount:", self.FeatureCount)
    if self.IsClassification:
      print("  |__ ClassCount:", self.ClassCount)
    
    if self.TSSamples is not None:
      print("  |__ Training set samples  : %d   shape:%s" % (self.TSSampleCount, self.TSSamples.shape) )
    if self.TSTargets is not None:
      print("  |__ Training set targets  : %d   shape:%s" % (self.TSSampleCount, self.TSTargets.shape) )
      
    if self.VSSamples is not None:
      print("  |__ Validation set samples: %d   shape:%s" % (self.VSSampleCount, self.VSSamples.shape) )
    if self.VSTargets is not None:
      print("  |__ Validation set targets: %d   shape:%s" % (self.VSSampleCount, self.VSTargets.shape) )
      
    if self.UTSamples is not None:
      print("  |__ Test set samples      : %d   shape:%s" % (self.UTSampleCount, self.UTSamples.shape) )
    if self.UTTargets is not None:
      print("  |__ Test set targets      : %d   shape:%s" % (self.UTSampleCount, self.UTTargets.shape) )
      
  # --------------------------------------------------------------------------------------
  def countSamples(self):
    if self.TSSamples is not None:
      self.TSSampleCount = self.TSSamples.shape[0]
      self.SampleCount = self.TSSampleCount + self.VSSampleCount
      
    if self.VSSamples is not None:
      self.VSSampleCount = self.VSSamples.shape[0]
      self.SampleCount = self.TSSampleCount + self.VSSampleCount         
      
    # The test set samples are not included in the available sample count
    if self.UTSamples is not None:
      self.UTSampleCount = self.UTSamples.shape[0]
  # --------------------------------------------------------------------------------------
  def Split(self, p_nTrainingPercentage):
    nTSSamples, nVSSamples, nTSTargets, nVSTargets = train_test_split(self.Samples, self.Targets
                                                            , test_size=1.0-p_nTrainingPercentage
                                                            , random_state=self.RandomSeed
                                                            , shuffle=True
                                                            , stratify=self.Targets
                                                            )
    self.TrainingSet(nTSSamples, nTSTargets)
    self.ValidationSet(nVSSamples, nVSTargets)
    self.countSamples()
  # --------------------------------------------------------------------------------------
  def LoadCache(self, p_oFileStore, p_sSamplesFilePrefix="Samples", p_sTargetsFilePrefix="Targets", p_bIsVerbose=False):
    bResult = p_oFileStore.Exists("%s.pkl" % p_sSamplesFilePrefix)
    
    if bResult:
      if p_bIsVerbose:
        print("Loading known data set ...")
      self.Samples    = p_oFileStore.Deserialize("%s.pkl" % p_sSamplesFilePrefix)
      self.Targets    = p_oFileStore.Deserialize("%s.pkl" % p_sTargetsFilePrefix)
    
      if p_bIsVerbose:
        print("Loading training set ...")      
      nTSSamples  = p_oFileStore.Deserialize("%s.TS.pkl" % p_sSamplesFilePrefix) 
      nTSTargets   = p_oFileStore.Deserialize("%s.TS.pkl" % p_sTargetsFilePrefix)
      self.TrainingSet(nTSSamples, nTSTargets)
      
      if p_bIsVerbose:
        print("Loading validation set ...")            
      nVSSamples  = p_oFileStore.Deserialize("%s.VS.pkl" % p_sSamplesFilePrefix) 
      nVSTargets  = p_oFileStore.Deserialize("%s.VS.pkl" % p_sTargetsFilePrefix)
      self.ValidationSet(nVSSamples, nVSTargets)
      
      if p_bIsVerbose:
        print("Loading unknown test data set ...")            
      nUTSamples  = p_oFileStore.Deserialize("%s.UT.pkl" % p_sSamplesFilePrefix)
      if nUTSamples is not None:
        nUTTargets   = p_oFileStore.Deserialize("%s.UT.pkl" % p_sTargetsFilePrefix)
        self.UnknownTestSet(nUTSamples, nUTTargets)
      
    return bResult
  # --------------------------------------------------------------------------------------    
  def SaveCache(self, p_oFileStore, p_sSamplesFilePrefix="Samples", p_sTargetsFilePrefix="Targets"):
    if self.Samples is not None:
      p_oFileStore.Serialize("%s.pkl" % p_sSamplesFilePrefix, self.Samples, True) 
      p_oFileStore.Serialize("%s.pkl" % p_sTargetsFilePrefix, self.Labels, True)
    
    p_oFileStore.Serialize("%s.TS.pkl" % p_sSamplesFilePrefix, self.TSSamples, True) 
    p_oFileStore.Serialize("%s.TS.pkl" % p_sTargetsFilePrefix, self.TSLabels, True)
    
    p_oFileStore.Serialize("%s.VS.pkl" % p_sSamplesFilePrefix, self.VSSamples, True) 
    p_oFileStore.Serialize("%s.VS.pkl" % p_sTargetsFilePrefix, self.VSLabels, True)
    
    if self.UTSamples is not None:
      p_oFileStore.Serialize("%s.UT.pkl" % p_sSamplesFilePrefix, self.UTSamples, True) 
      p_oFileStore.Serialize("%s.UT.pkl" % p_sTargetsFilePrefix, self.UTLabels, True)
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
  ml.RandomSeed(2023)
  
  
  #nData = np.random.rand(100).astype(np.float32)
  nData = np.arange(0, 100)
  nData = nData / (nData + 1)
  print(nData.shape)
  nTargets = np.concatenate([np.zeros(50), np.ones(50)], axis=0).astype(np.int32)
  print(nTargets.shape)
  
  oDataset = CDataSetBase("dummy")
  oDataset.Samples = nData
  oDataset.Targets = nTargets
  oDataset.Split(0.8)
  oDataset.PrintInfo()
  
  nTSChecksum = np.sum(oDataset.TSSamples)
  nVSChecksum = np.sum(oDataset.VSSamples)
  print(nTSChecksum, nVSChecksum)
  
  # // Unit Testing \\
  assert nTSChecksum == 75.59110064555725
  assert nVSChecksum == 19.22152183680314
  
  # Seed None:                 75.45033306926038 19.362289413100005
  # Seed None: Parameter 2023: 75.59110064555725 19.22152183680314
  # Seed 2023: Parameter 2023: 75.59110064555725 19.22152183680314  # First call with the same seed
  # Seed 2023: Parameter None: 75.59110064555725 19.22152183680314  # First call with the same seed
  # Seed 2025:                 75.55118094349115 19.261441538869224
  
  
  
