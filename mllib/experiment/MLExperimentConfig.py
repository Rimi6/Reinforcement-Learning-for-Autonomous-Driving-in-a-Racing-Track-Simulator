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

import os
import json


# =========================================================================================================================
class CMLExperimentConfig(dict):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_sFileName, p_nExperimentNumber=None, p_nModelVariation=None):
    self.ModelVariation = p_nModelVariation
    
    # reading the data from the file
    if os.path.exists(p_sFileName):
      with open(p_sFileName) as oFile:
        sConfig = oFile.read()
        
        self.setDefaults()        
      
        dData = json.loads(sConfig)
    
      for sKey in dData.keys():
        self[sKey] = dData[sKey]
    
    if p_nExperimentNumber is not None:
      self["ExperimentNumber"] = p_nExperimentNumber
    if self.ModelVariation is not None:
      self["ModelVariation"] = self.ModelVariation
  # --------------------------------------------------------------------------------------    
  @property
  def ExperimentCode(self):
    sCode = self["ModelName"] 
    if "ModelVariation" in self:
      sCode += "_" + self["ModelVariation"]
    if "ExperimentNumber" in self:
      sCode = sCode + "_%02d" % self["ExperimentNumber"]
    return sCode
  # --------------------------------------------------------------------------------------    
  def setDefaults(self):
    pass
  # --------------------------------------------------------------------------------------
  @classmethod
  def Load(cls, p_sFileName):    
    from mllib.system import CMLSystem
    oConfig = CMLExperimentConfig(CMLSystem.Instance().ModelFS.File(p_sFileName))
    return oConfig  
  # --------------------------------------------------------------------------------------        
# =========================================================================================================================        


  