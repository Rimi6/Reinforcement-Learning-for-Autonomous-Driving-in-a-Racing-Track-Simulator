# ......................................................................................
# MIT License

# Copyright (c) 2021-2023 Pantelis I. Kaplanoglou

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
import glob
import json

import sys
if (sys.version_info.major == 3) and (sys.version_info.minor <= 7):
  import pickle5 as pickle
else:  
  import pickle  

# =======================================================================================================================
class CFileStore(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_sBaseFolder, p_bIsVerbose=False):
    #.......................... |  Instance Attributes | ............................
    self.BaseFolder = p_sBaseFolder
    self.IsVerbose  = p_bIsVerbose
    #................................................................................
    if not os.path.exists(self.BaseFolder):
      os.makedirs(self.BaseFolder)
  # --------------------------------------------------------------------------------------------------------
  @property
  def HasData(self):
    bResult = os.path.exists(self.BaseFolder)
    if bResult:
      oFiles = os.listdir(self.BaseFolder)
      nFileCount = len(oFiles)
      bResult = nFileCount > 0

    return bResult;
  # --------------------------------------------------------------------------------------------------------
  def Exists(self, p_sFileName):
    sFullFilePath = os.path.join(self.BaseFolder, p_sFileName)
    return os.path.isfile(sFullFilePath)
  # --------------------------------------------------------------------------------------------------------
  def SubFS(self, p_sSubFolderName):
    return CFileStore(self.SubPath(p_sSubFolderName))
  # --------------------------------------------------------------------------------------------------------
  def Files(self, p_sFileMatchingPattern, p_bIsRemovingExtension=False, p_tSortFilenamesAs=None):
    sEntries = glob.glob1(self.BaseFolder, p_sFileMatchingPattern)
    if p_bIsRemovingExtension:
      p_tSortFilenamesAs
      oFileNamesOnly = []
      for sEntry in sEntries:
        sFileNameOnly, _ = os.path.splitext(sEntry)
        oFileNamesOnly.append(sFileNameOnly)
      sEntries = sorted(oFileNamesOnly, key=p_tSortFilenamesAs)
    
    oResult = [os.path.join(self.BaseFolder, x) for x in sEntries]
     
    return oResult  
  # --------------------------------------------------------------------------------------------------------
  @property
  def DirectoryEntries(self):
    return os.listdir(self.BaseFolder)
  # --------------------------------------------------------------------------------------------------------
  @property
  def IsEmpty(self):
    return (len(os.listdir(self.BaseFolder)) == 0)
  # --------------------------------------------------------------------------------------------------------
  def SubPath(self, p_sSubPath):
    if os.path.sep == "\\":
      if p_sSubPath.find(os.path.sep) < 0:
        p_sSubPath = p_sSubPath.replace("/", "\\")
    
    return self.Folder(p_sSubPath)
  # --------------------------------------------------------------------------------------------------------
  def Folder(self, p_sSubPath):
    sFolder = os.path.join(self.BaseFolder, p_sSubPath)    
    if not os.path.exists(sFolder):
      os.makedirs(sFolder)
      
    return sFolder    
  # --------------------------------------------------------------------------------------------------------
  def File(self, p_sFileName, p_sFileExt=None):
    if p_sFileExt is not None:
      p_sFileName += p_sFileExt
    return os.path.join(self.BaseFolder, p_sFileName)    
  # --------------------------------------------------------------------------------------------------------
  def Deserialize(self, p_sFileName, p_bIsPython2Format=False, p_sErrorTemplate=None):
    """
    Deserializes the data from a pickle file if it exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        The object with its data or None when the file is not found.
    """
    oData=None
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
       
    if os.path.isfile(p_sFileName):
      if self.IsVerbose :
        print("      {.} Loading data from %s" % p_sFileName)

      with open(p_sFileName, "rb") as oFile:
        if p_bIsPython2Format:
          oUnpickler = pickle._Unpickler(oFile)
          oUnpickler.encoding = 'latin1'
          oData = oUnpickler.load()
        else:
          oData = pickle.load(oFile)
        oFile.close()
    else:
      if p_sErrorTemplate is not None:
        raise Exception(p_sErrorTemplate % p_sFileName)
       
    return oData
  #----------------------------------------------------------------------------------
  def WriteTextToFile(self, p_sFileName, p_sText):
    """
    Writes text to a file

    Parameters
        p_sFileName        : Full path to the text file
        p_sText            : Text to write
    """
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
    
    if self.IsVerbose :
      print("  {.} Saving text to %s" % p_sFileName)

    if isinstance(p_sText, list):
      with open(p_sFileName, "w") as oFile:
        for sLine in p_sText:
          print(sLine, file=oFile)
        oFile.close()      
    else:
      with open(p_sFileName, "w") as oFile:
        print(p_sText, file=oFile)
        oFile.close()

    return True
  #----------------------------------------------------------------------------------
  def Serialize(self, p_sFileName, p_oData, p_bIsOverwritting=False, p_sExtraDisplayLabel=None):
    """
    Serializes the data to a pickle file if it does not exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        True if a new file was created
    """
    bResult=False
    
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)

    if p_bIsOverwritting:
      bMustContinue = True
    else:
      bMustContinue = not os.path.isfile(p_sFileName)
        
    if bMustContinue:
      if self.IsVerbose :
        if p_sExtraDisplayLabel is not None:
            print("  {%s} Saving data to %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
        else:
            print("  {.} Saving data to %s" % p_sFileName)
      with open(p_sFileName, "wb") as oFile:
          pickle.dump(p_oData, oFile, pickle.HIGHEST_PROTOCOL)
          oFile.close()
      bResult=True
    else:
      if self.IsVerbose:
          if p_sExtraDisplayLabel is not None:
              print("  {%s} Not overwritting %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
          else:
              print("  {.} Not overwritting %s" % p_sFileName)
                            
    return bResult
  #----------------------------------------------------------------------------------
  def LoadJSON(self, p_sFileName, p_sErrorTemplate=None):
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
          
    dResult = None
    if os.path.exists(p_sFileName):
      with open(p_sFileName) as oFile:
        sJSON = oFile.read()
      dResult = json.loads(sJSON)
    else:
      if p_sErrorTemplate is not None:
        raise Exception(p_sErrorTemplate % p_sFileName)
      
    return dResult  
  #----------------------------------------------------------------------------------
  def SaveJSON(self, p_sFileName, p_oObject):
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
          
    sJSON = json.dumps(p_oObject, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    with open(p_sFileName, "w") as oFile:
      oFile.write(sJSON)
      oFile.close()
  #----------------------------------------------------------------------------------
  def __repr__(self)->str:
    return self.BaseFolder
  #----------------------------------------------------------------------------------    
  def __str__(self)->str:
    return self.BaseFolder
  #----------------------------------------------------------------------------------
# =======================================================================================================================


if __name__ == "__main__":
  oFS = CFileStore("T:\MLModels.Keep\REXPLAINET_MNIST_16.10\checkpoints")
  sFiles = oFS.Files("*.index", True, p_tSortFilenamesAs=int)
  print("\n".join(sFiles))
  
  if False:
    oFS = CFileStore("MLData")
    print(oFS.SubPath("test/test2"))
    print(oFS.Folder("subfolder"))
    print(oFS.File("test"))
    print(oFS.SubFS("CIFAR10").File("test"))


