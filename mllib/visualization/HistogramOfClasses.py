import matplotlib.pyplot as plt
import numpy as np


# =========================================================================================================================
class CHistogramOfClasses(object):  # class CPlot: object
  # --------------------------------------------------------------------------------------
  def __init__(self, p_nData, p_nClasses, p_bIsProbabilities=False):
    self.Data = p_nData
    self.Classes = p_nClasses
    self.IsProbabilities = p_bIsProbabilities
  # --------------------------------------------------------------------------------------
  def Show(self):

    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.hist(self.Data, density=self.IsProbabilities, bins=self.Classes, ec="k") 
    ax.locator_params(axis='x', integer=True)

    if self.IsProbabilities:
      plt.ylabel('Probabilities')
    else:
      plt.ylabel('Counts')
    plt.xlabel('Classes')
    plt.show()
  # --------------------------------------------------------------------------------------


# =========================================================================================================================