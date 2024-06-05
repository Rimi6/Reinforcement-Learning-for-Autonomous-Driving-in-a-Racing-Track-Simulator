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


# =========================================================================================================================
class CPlotConfusionMatrix(object):

    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfusionMatrix):
        self.ConfusionMatrix = p_oConfusionMatrix

    # --------------------------------------------------------------------------------------
    def Show(self):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(self.ConfusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(self.ConfusionMatrix.shape[0]):
            for j in range(self.ConfusionMatrix.shape[1]):
                ax.text(x=j, y=i, s=self.ConfusionMatrix[i, j], va='center', ha='center', size='xx-large')
         
        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('Actual Label'   , fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
    # --------------------------------------------------------------------------------------
# =========================================================================================================================    

