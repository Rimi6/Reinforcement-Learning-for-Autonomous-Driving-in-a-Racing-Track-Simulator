import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class CClassColorMap(object):
  def __init__(self, p_sColorNames=["black", "red", "green", "blue", "yellow"]):
    self.ColorNames = p_sColorNames
    self.ColorCount = len(self.ColorNames)
  
 
  def Make(self):
    self.ColorCount = len(self.ColorNames)
    
    # Map a class number to a color
    nRange = range(self.ColorCount)
    oDict = {   nIndex: colors.to_rgb(self.ColorNames[nIndex]) for nIndex in nRange }
    
    # Create a colormap (optional)
    oColorRGB = [oDict[nIndex] for nIndex in nRange]
    oColorMap = colors.ListedColormap(oColorRGB)
    #oMyColorMapNorm = colors.BoundaryNorm(np.arange(C_p + 1) - 0.5, C_p)
    oColorMapNorm = colors.BoundaryNorm(np.asarray(range(self.ColorCount + 1), np.float32)
                                        , self.ColorCount)
    return oColorMap, oColorMapNorm
    
  