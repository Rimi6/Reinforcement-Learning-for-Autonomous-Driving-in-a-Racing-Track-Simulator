import matplotlib.pyplot as plt


class CAutoMultiImagePlot(object):
  def __init__(self):
    self.Rows    = [] 
    self.RowCount = 0
    self.CurrentRow = -1
    self.RowColCount = dict()
    self.MaxColCount = 0
    
  def AddRow(self):
    self.CurrentRow = self.RowCount
    self.Rows.append([])
    self.RowCount = len(self.Rows)
    
  def AddColumn(self, p_nImage, p_sTitle=None):
    oRowColumns = self.Rows[self.CurrentRow]
    dImage = {"image": p_nImage, "title": p_sTitle}
    
    oRowColumns.append(dImage)
    self.Rows[self.CurrentRow] = oRowColumns
    nColCount = len(oRowColumns)
    
    
    self.RowColCount[self.CurrentRow] = nColCount
    if nColCount > self.MaxColCount:
      self.MaxColCount = nColCount
  
  def Show(self, p_sTitle=None, p_nFigureSize=(15, 6)):
    
    
    fig, oSubplotGrid = plt.subplots(nrows=self.RowCount, ncols=self.MaxColCount
                                    , figsize=p_nFigureSize, subplot_kw={'xticks': [], 'yticks': []})
    plt.title(p_sTitle)
    if self.RowCount == 1:
      for nColIndex,dImage in enumerate(self.Rows[0]):
        oSubPlot = oSubplotGrid[nColIndex]
        oSubPlot.title.set_text(dImage["title"])
        oSubPlot.imshow(dImage["image"]) 
    else:
      for nRowIndex,oRowColumns in enumerate(self.Rows):
        for nColIndex,dImage in enumerate(oRowColumns):
          oSubPlot = oSubplotGrid[nRowIndex, nColIndex]
          oSubPlot.title.set_text(dImage["title"])
          oSubPlot.imshow(dImage["image"]) 

    plt.show()
      
