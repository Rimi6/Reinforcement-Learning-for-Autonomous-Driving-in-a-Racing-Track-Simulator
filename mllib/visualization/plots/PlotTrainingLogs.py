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
class CPlotTrainingLogs(object):

    # --------------------------------------------------------------------------------------
    def __init__(self, p_dTrainingLog):
        self.TrainingLog = p_dTrainingLog
        print("Keys of training process log:", self.TrainingLog.keys())
        
    # --------------------------------------------------------------------------------------
    def Show(self, p_sModelName, p_sMetricName="accuracy", p_sCostFunctionName="cost", p_oCostFunction=None):
        sValMetricName = "val_" + p_sMetricName
        
        # Plot the accuracy during the training epochs
        plt.plot(self.TrainingLog[p_sMetricName])
        if sValMetricName in self.TrainingLog:
          plt.plot(self.TrainingLog[sValMetricName])
        plt.title(p_sModelName + ' ' + p_sMetricName)
        plt.ylabel(p_sMetricName)
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

        # Plot the error during the training epochs
        sCostFunctionName = p_sCostFunctionName
        if (p_sCostFunctionName is None) and (p_oCostFunction is not None):
            sCostFunctionNameParts = p_oCostFunction.name.split("_")  # [PYTHON]: Splitting string into an array of strings
            sCostFunctionNameParts = [x.capitalize() + " " for x in
                                      sCostFunctionNameParts]  # [PYTHON]: List comprehension example
            sCostFunctionName = " ".join(
                sCostFunctionNameParts)  # [PYTHON]: Joining string in a list with the space between them

        plt.plot(self.TrainingLog["loss"])
        if "val_loss" in self.TrainingLog:
          plt.plot(self.TrainingLog["val_loss"])
        plt.title(p_sModelName + ' ' + sCostFunctionName)
        plt.ylabel("Error (Cost)")
        plt.xlabel("Epoch")
        plt.legend(["train", 'validation'], loc='upper left')
        plt.show()
    # --------------------------------------------------------------------------------------
# =========================================================================================================================

