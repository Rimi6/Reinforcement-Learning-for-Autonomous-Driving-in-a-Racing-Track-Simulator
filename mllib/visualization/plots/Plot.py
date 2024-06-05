# ......................................................................................
# MIT License

# Copyright (c) 2020-2023 Pantelis I. Kaplanoglou

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

from matplotlib import colors           

import matplotlib.pyplot as plt  # use the subpackage (a.k.a. namespace) with the alias "plt"
import numpy as np


# ====================================================================================================
class CPlot(object):  # class CPlot: object

    # --------------------------------------------------------------------------------------
    # Constructor
    def __init__(self, p_sTitle, p_oSamples, p_oLabels):
        # ................................................................
        # // Fields \\
        self.Title = p_sTitle
        self.Samples = p_oSamples
        self.Labels = p_oLabels

        # ................................................................
    # --------------------------------------------------------------------------------------
    def Show(self, p_bIsMinMaxScaled=False, p_nLineSlope=None, p_nLineIntercept=None, p_nLimitsX=[-4, 4], p_nLimitsY=[-4, 4]):

        # Two dimensional data for the scatter plot
        nXValues = self.Samples[:, 0]
        nYValues = self.Samples[:, 1]
        nLabels = self.Labels
        
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        oColors = ["darkorange", "darkseagreen"]
        oLabelDescriptions = ["orange tree", "olive tree"]
        oColorMap = colors.ListedColormap(oColors)
    
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.scatter(nXValues, nYValues, c=nLabels, cmap=oColorMap)
    
        plt.title(self.Title)
        cb = plt.colorbar()
        nLoc = np.arange(0, max(nLabels), max(nLabels) / float(len(oColors)))
        cb.set_ticks(nLoc)
        cb.set_ticklabels(oLabelDescriptions)

        if (p_nLineSlope is not None):
            x1 = np.min(nXValues)
            y1 = p_nLineSlope * x1 + p_nLineIntercept;
            x2 = np.max(nXValues)
            y2 = p_nLineSlope * x2 + p_nLineIntercept;
            oPlot1 = ax.plot([x1, x2], [y1, y2], 'r--', label="Decision line")
            oLegend = plt.legend(loc="upper left", shadow=True, fontsize='x-large')
            oLegend.get_frame().set_facecolor("lightyellow")
       
        if p_bIsMinMaxScaled:
            ax.set_xlim((-0.05, 1.05))
            ax.set_ylim((-0.05, 1.05))
        else:
            ax.set_xlim(p_nLimitsX[0], p_nLimitsX[1])
            ax.set_ylim(p_nLimitsY[0], p_nLimitsY[1])  
            
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # plt.scatter(oDataset.Samples[:,0], oDataset.Samples[:,1])
        # , t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

        plt.show()
    # --------------------------------------------------------------------------------------
# ====================================================================================================
