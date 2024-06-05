import numpy as np
import matplotlib.pyplot as plt        
from matplotlib import colors           
from matplotlib import cm
from scipy import interpolate

# =========================================================================================================================
class CMultiImagePlot(object): 
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_oPlotDimensions=(11,6)):
    # ................................................................
    self.Clear()
    self.PlotDimensions = p_oPlotDimensions
    # ................................................................
  # --------------------------------------------------------------------------------------
  def Clear(self):
    self.Images = []
    self.ColorMaps = []
    self.BoundaryNorms = []
    self.Rows = None
    self.Columns = None
    self.CurrentRowIndex = -1
  # --------------------------------------------------------------------------------------
  def AddRow(self):
    self.Images.append([])
    self.ColorMaps.append([])
    self.BoundaryNorms.append([])
    self.CurrentRowIndex += 1
  # --------------------------------------------------------------------------------------
  def AddImageColumn(self, p_nImage, p_sColorMap=None, p_nBoundaryNorm=None):
    oRow = self.Images[self.CurrentRowIndex]
    oRow.append(p_nImage)
    self.ColorMaps[self.CurrentRowIndex].append(p_sColorMap)
    self.BoundaryNorms[self.CurrentRowIndex].append(p_nBoundaryNorm)
  # --------------------------------------------------------------------------------------
  def inferGridDimensions(self):
    self.Rows = len(self.Images)
    self.Columns = 0
    for oImageList in self.Images:
      if len(oImageList) > self.Columns:
        self.Columns = len(oImageList)
  # --------------------------------------------------------------------------------------
  def Show(self, p_bTranspose=True, p_nMin=None, p_nMax=None):
    self.inferGridDimensions()
    
    if p_bTranspose:
      nRows    = len(self.Images)
      nColumns = len(self.Images[0])
          
      fig11 = plt.figure(figsize=self.PlotDimensions, constrained_layout=True)
      outer_grid = fig11.add_gridspec(nColumns, nRows, wspace=0, hspace=2)
  
                                  
      for nRowIndex, oRows in enumerate(self.Images):
        for nColIndex, nImage in enumerate(oRows):
          
          sColorMap = self.ColorMaps[nRowIndex][nColIndex]
          nBoundaryNorm = self.BoundaryNorms[nRowIndex][nColIndex]
          # gridspec inside gridspec
          
          #inner_grid = outer_grid[nColIndex, nRowIndex]
          inner_grid = outer_grid[nColIndex, nRowIndex].subgridspec(1,1, wspace=0, hspace=0)
          axs = inner_grid.subplots()  # Create all subplots for the inner grid.
          
          axs.set(xticks=[], yticks=[])
          if sColorMap is None:
            axs.imshow(nImage, interpolation=None, vmin=p_nMin, vmax=p_nMax)
          else:
            axs.imshow(nImage, cmap=sColorMap, norm=nBoundaryNorm, interpolation=None, vmin=p_nMin, vmax=p_nMax)
    else:
      fig11 = plt.figure(figsize=self.PlotDimensions, constrained_layout=True)
      outer_grid = fig11.add_gridspec(self.Rows, self.Columns, wspace=0, hspace=2)
          
                                  
      for nRowIndex, oRowImages in enumerate(self.Images):
        for nColIndex, nImage in enumerate(oRowImages):
          
          sColorMap     = self.ColorMaps[nRowIndex][nColIndex]
          nBoundaryNorm = self.BoundaryNorms[nRowIndex][nColIndex]
          # gridspec inside gridspec
          
          #inner_grid = outer_grid[nColIndex, nRowIndex]
          inner_grid = outer_grid[nRowIndex, nColIndex].subgridspec(1,1, wspace=0, hspace=0)
          axs = inner_grid.subplots()  # Create all subplots for the inner grid.
          
          axs.set(xticks=[], yticks=[])
          if sColorMap is None:
            axs.imshow(nImage, interpolation=None, vmin=p_nMin, vmax=p_nMax)
          else:
            axs.imshow(nImage, cmap=sColorMap, norm=nBoundaryNorm, interpolation=None, vmin=p_nMin, vmax=p_nMax)      
    
    #plt.tight_layout(pad=1.01)
    plt.show()        
  # --------------------------------------------------------------------------------------      
  def Show_Legacy(self):
    nRows    = len(self.Images)
    nColumns = self.ColumnsPerRow
    
    self.__fig, self.__ax = plt.subplots(nrows=nColumns, ncols=nRows, sharex=False, sharey=False, squeeze=True)#, figsize=self.PlotDimensions)
    # remove the x and y ticks
    for ax in self.__ax :
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    for nRowIndex, oRows in enumerate(self.Images):
      for nColIndex, nImage in enumerate(oRows):
        sColorMap = self.ColorMaps[nRowIndex][nColIndex]
        #nRow = p_nIndex // self.PanesPerRow
        #nCol = p_nIndex - (nRow * self.PanesPerRow)
        oPlot = self.__ax[nColIndex, nRowIndex]
        #oPlot.axis("off")
        #oPlot.set_xlabel("")
        #oPlot.set_ylabel("")
        oPlot.set_yticklabels([])
        oPlot.set_xticklabels([])
        
        
        oPlot.set_title("%d %d" % (nRowIndex, nColIndex))
        if sColorMap is None:
          oPlot.imshow(nImage, interpolation=None)
        else:
          oPlot.imshow(nImage, cmap=sColorMap, interpolation=None)
        #oPlot.set_xlim([0, 32])
        #oPlot.set_ylim([0, 32])
        #oPlot.set_frame_on(False)
                  

    
    plt.tight_layout(pad=1.01)
    plt.show()
    
  # --------------------------------------------------------------------------------------
# =========================================================================================================================