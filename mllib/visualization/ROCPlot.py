from sklearn import metrics
import matplotlib.pyplot as plt

# =========================================================================================================================
'''
    Receiver Operator Characteristics (ROC) Plot
'''
class CROCPlot(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_nTrueLabels, p_nPredictedProbs):
    self.Actual         = p_nTrueLabels
    self.PredictedProbs = p_nPredictedProbs
  # --------------------------------------------------------------------------------------
  def Show(self, p_nTrueThreshold=0.5, p_nFigureSize=[6.00, 6.00], p_bIsShowingGrid=True):
    plt.rcParams["figure.figsize"]    = p_nFigureSize
    plt.rcParams["figure.autolayout"] = True
    
    nFPR, nTPR, nThresholds  = metrics.roc_curve(self.Actual,  self.PredictedProbs)
    nAUC = metrics.roc_auc_score(self.Actual, self.PredictedProbs)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    if p_bIsShowingGrid:
      plt.grid()
    plt.plot(nFPR, nTPR, label="ROC curve (AUC=%.2f)" % nAUC, linewidth=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', label="Random prediction")#, color="yellow", linewidth=1)
    
    plt.ylabel("TPR (True Positive Rate")
    plt.xlabel("FPR (False Positive Rate)")
    
    for nThesholdIndex,nThreshold in enumerate(nThresholds):
      if nThreshold < p_nTrueThreshold:
        break
        
    plt.plot(nFPR[nThesholdIndex], nTPR[nThesholdIndex] 
              ,label="Threshold %.2f" % nThreshold
              ,marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="yellow")
    plt.legend(loc=4)
    plt.show()
  # --------------------------------------------------------------------------------------
# =========================================================================================================================      