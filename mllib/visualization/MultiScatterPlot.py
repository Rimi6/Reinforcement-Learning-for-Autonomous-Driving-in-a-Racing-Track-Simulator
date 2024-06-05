from matplotlib import cm
from matplotlib import colors           

import matplotlib.pyplot as plt        
import numpy as np


# =========================================================================================================================
class CMultiScatterPlot(object): 
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_sTitle, p_oData=None, p_nClassCount=10, p_sClassNames=None):
    # ................................................................
    # // Fields \\
    self.Title    = p_sTitle
    self.Data  = p_oData

    self.ClassCount = p_nClassCount
    if p_sClassNames is not None:
      self.ClassNames  = p_sClassNames
      self.ClassCount = len(self.ClassNames)

    if (self.ClassCount <= 10):
      self.ColorMap   = cm.get_cmap("tab10")
    elif (self.ClassCount <= 20):
      self.ColorMap   = cm.get_cmap("tab20")
    else:
      self.ColorMap = cm.get_cmap("prism")
      #self.ColorMap = colors.ListedColormap(["darkorange","darkseagreen"])
    
    self.PlotDimensions = [14,8]
    self.PointSize = 48
    
    self.PanesPerRow  = 2
    if self.Data is not None:
      self.Panes        = len(self.Data)

    self.__fig = None
    self.__ax  = None
    # ................................................................
  # --------------------------------------------------------------------------------------
  def AddData(self, p_sDataName, p_nSamples, p_nLabels=None):
    if self.Data is None:
      self.Data = []

    if p_nLabels is None:
      p_nLabels = np.zeros((p_nSamples.shape[0]), np.int32)
    self.Data.append([p_sDataName, p_nSamples, p_nLabels])
    self.Panes = len(self.Data)
  # --------------------------------------------------------------------------------------
  def Show(self, p_nIndex, p_sXLabel, p_sYLabel):

    # Two dimensional data for the scatter plot
    sDataName, nSamples, nLabels = self.Data[p_nIndex]
    nXValues = nSamples[:,0]
    nYValues = nSamples[:,1]
    nLabels  = nLabels

    if self.__fig is None:
      if self.Panes == 1:
        self.__fig, self.__ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=False, figsize=self.PlotDimensions)
      else:
        nRows = self.Panes // self.PanesPerRow
        if (self.Panes % self.PanesPerRow !=  0):
          nRows += 1
        self.__fig, self.__ax = plt.subplots(nrows=nRows, ncols=self.PanesPerRow, sharex=False, sharey=False, squeeze=False, figsize=self.PlotDimensions)

      

    nRow = p_nIndex // self.PanesPerRow
    nCol = p_nIndex - (nRow * self.PanesPerRow)
    oPlot = self.__ax[nCol, nRow]
    oPlot.set_xlabel(p_sXLabel)
    oPlot.set_ylabel(p_sYLabel)
    oPlot.set_title(sDataName)



    oScatter = oPlot.scatter(nXValues, nYValues, s=self.PointSize, c=nLabels, cmap=self.ColorMap)
    
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
    # produce a legend with the unique colors from the scatter
    oLegend = oPlot.legend(*oScatter.legend_elements(), loc="lower right", title="Classes", framealpha=0.4, labelspacing=0.1)
    oPlot.add_artist(oLegend)


    #if self.ClassNames is not None:
    #  cb = plt.colorbar()
    #  nLoc = np.arange(0,max(nLabels),max(nLabels)/float(len(oColors)))
    #  cb.set_ticks(nLoc)
    #  cb.set_ticklabels(oLabelDescriptions)


    #if (line is not None):
    #  lineSlope         = line[0]  
    #  lineDisplacement  = line[1]
    #  x1 = np.min(nXValues)
    #  y1 = lineSlope * x1 + lineDisplacement;
    #  x2 = np.max(nXValues)
    #  y2 = lineSlope * x2 + lineDisplacement;
    #  oPlot1 = ax.plot([x1,x2], [y1,y2], 'r--', label="Decision line")
    #  oLegend = plt.legend(loc = "upper left", shadow=True, fontsize='x-large')
    #  oLegend.get_frame().set_facecolor("lightyellow")


       
    #if p_bIsMinMaxScaled:
    #  ax.set_xlim( (0.0, 1.0) )
    #  ax.set_ylim( (0.0, 1.0) )

    #ax.legend(loc="upper left")

    #plt.scatter(oDataset.Samples[:,0], oDataset.Samples[:,1])
              #, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    
    if p_nIndex == self.Panes - 1:
      plt.title(self.Title)
      plt.tight_layout(pad=1.01)
      plt.show()
  # --------------------------------------------------------------------------------------
# =========================================================================================================================