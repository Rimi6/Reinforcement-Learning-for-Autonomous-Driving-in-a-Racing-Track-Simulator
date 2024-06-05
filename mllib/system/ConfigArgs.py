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

# =======================================================================================================================
class CConfigArgs(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_oJSON=None):
    if p_oJSON is not None:
      if isinstance(p_oJSON, dict):
        self.__dict__ = p_oJSON
      else:
        self.__dict__ = json.loads(p_oJSON)
  # --------------------------------------------------------------------------------------------------------
  @classmethod
  def LoadJSON(cls, p_sFileName):
    oResult = None
        
    if os.path.exists(p_sFileName):
      with open(p_sFileName) as oFile:
        sJSON = oFile.read()
      dConfig = json.loads(sJSON)
      oResult = cls(dConfig)
      
    return oResult
  # --------------------------------------------------------------------------------------------------------
  def ToJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
  # --------------------------------------------------------------------------------------------------------  
# =======================================================================================================================
