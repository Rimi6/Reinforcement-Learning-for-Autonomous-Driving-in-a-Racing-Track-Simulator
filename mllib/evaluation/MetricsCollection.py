# ......................................................................................
# MIT License

# Copyright (c) 2020-2022 Pantelis I. Kaplanoglou

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
import numpy as np
from datetime import datetime
from mllib.evaluation import CMetric
from mllib.classes import CAutoObject


#==============================================================================================================================
class CMetricsPacket(CAutoObject):
  #--------------------------------------------------------------------------------------------------------------
  def __init__(self, p_sNames, p_nValues):
  
    dObjectFields = dict()
    for nIndex, sName in enumerate(p_sNames):
      dObjectFields[sName] = p_nValues[nIndex]
      
    super(CMetricsPacket, self).__init__(dObjectFields)
  #--------------------------------------------------------------------------------------------------------------
#==============================================================================================================================





#==============================================================================================================================
class CMetricsCollection(dict):
  #--------------------------------------------------------------------------------------------------------------
  def __init__(self, p_sMetricNamePrefix=None, p_nPeriodSteps=1):
    self.MetricNamePrefix   = p_sMetricNamePrefix
    self.FunctionMapping    = dict()
        
    self.Series             = dict()
    self.PeriodsDurations   = []
    self.Values             = None
    self.Previous           = None
    self.Current            = None
    self.__currentMetrics   = None
    self.__stepCount          = 0
    self.__isRunning        = False
    self.__periodStartTime  = None
    self.__periodEndTime    = None
    
    self.PeriodSteps        = p_nPeriodSteps
  #--------------------------------------------------------------------------------------------------------------
  @property
  def StepCount(self):
    return self.__stepCount
  #--------------------------------------------------------------------------------------------------------------
  @property
  def PeriodDuration(self):
    if self.__periodEndTime is None:
      return datetime.now() - self.__periodStartTime
    else:
      return self.__periodEndTime - self.__periodStartTime
  #--------------------------------------------------------------------------------------------------------------
  def Register(self, p_sNames):
    if not isinstance(p_sNames, list):
      p_sNames = [p_sNames]
      
    oMetrics = []
    for sName in p_sNames:
      oMetric = CMetric(sName)
      self.AddMetric(oMetric)
      oMetrics.append(oMetric)
      
    if len(oMetrics) == 1:
      return oMetrics[0]
    else:
      return tuple(oMetrics)
  #--------------------------------------------------------------------------------------------------------------  
  def RegisterFunction(self, p_sNames, p_fMetric):
    if not isinstance(p_sNames, list):
      p_sNames = [p_sNames]
      
    oMetrics = []
    for sName in p_sNames:
      oMetric = CMetric(sName, p_fMetric)
      self.AddMetric(oMetric)   
      oMetrics.append(oMetric)
    if len(oMetrics) == 1:
      return oMetrics[0]
    else:
      return tuple(oMetrics)
  #--------------------------------------------------------------------------------------------------------------    
  def Add(self, p_oMetricList):
    for oMetric in p_oMetricList:
      self.AddMetric(oMetric)
    return self
  #--------------------------------------------------------------------------------------------------------------
  def AddMetric(self, p_oMetric):
    p_oMetric.Index = len(self.keys())
    self[p_oMetric.Name]        = p_oMetric
    self.Series[p_oMetric.Name] = []
     
    sFunctionName = p_oMetric.FunctionName
    if sFunctionName not in self.FunctionMapping:
      self.FunctionMapping[sFunctionName] = [p_oMetric.Function]
    self.FunctionMapping[sFunctionName].append(p_oMetric)
    return self
  #--------------------------------------------------------------------------------------------------------------
  def Clear(self):
    for sKey in self:
      self.Series[sKey] = []
  #--------------------------------------------------------------------------------------------------------------
  def Restore(self):
    sKeys = list(self.keys())
    for nIndex, nSeries in enumerate(self.Values):
      sMetricName = sKeys[nIndex]
      self.Series[sMetricName] = nSeries
  #--------------------------------------------------------------------------------------------------------------
  @property
  def IsPeriod(self):
    return (self.__stepCount % self.PeriodSteps) == 0
  #--------------------------------------------------------------------------------------------------------------
  @property
  def LastPeriodFinished(self):
    return (self.__stepCount % self.PeriodSteps) <= 1
  #--------------------------------------------------------------------------------------------------------------
  def BeginPeriod(self):
    if not self.__isRunning:
      self.__isRunning = True
      self.__stepCount = 0
      self.__periodStartTime = datetime.now()
      self.Previous  = self.Current
      self.Current   = None
    return self
  #--------------------------------------------------------------------------------------------------------------
  def NextStep(self):
    # TODO Step and epoch statistics
    self.__stepCount += 1
    return self      
  #--------------------------------------------------------------------------------------------------------------
  def EndPeriod(self):
    if self.__isRunning:
      self.__isRunning = True
      self.__periodEndTime = datetime.now()
      self.PeriodsDurations.append(self.PeriodDuration)
    return self    
  #--------------------------------------------------------------------------------------------------------------
  def Calculate(self, *args):
    for sFunction in self.FunctionMapping:
      oMapping = self.FunctionMapping[sFunction]
      func = oMapping[0]
      oResult = func(*args)
      if oResult is not None:
        for nIndex, oMetric in enumerate(oMapping[1:]):
          oSeries = self.Series[oMetric.Name]
          oValue = oResult[nIndex]
          if isinstance(oValue, np.ndarray):
            oSeries += oValue.tolist()
          else:
            oSeries.append(oValue)
    
    return self
  #--------------------------------------------------------------------------------------------------------------
  def Collect(self, p_sNames, p_oValues):
    if not isinstance(p_sNames, list):
      p_sNames = [p_sNames]
      p_oValues = [p_oValues]
      
    for nIndex, sName in enumerate(p_sNames):
      oValue = p_oValues[nIndex]
      oSeries = self.Series[sName]
      if isinstance(oValue, np.ndarray):
        oValues = oValue.tolist()
        if not isinstance(oValues, list):
          oSeries.append(oValues)
        else:
          oSeries += oValues 
      else:
        oSeries.append(oValue)
    
    return self
  #--------------------------------------------------------------------------------------------------------------
  def Aggregate(self,p_bIsClearing=False):
    # Ensure a finished metrics collection period
    self.Values = []
    self.__currentMetrics = []
    for sMetricName in self:
      oMetric = self[sMetricName]
      if len(self.Series[sMetricName]) == 0:
        self.__currentMetrics.append(0)
        self.Values.append(None)
      else:
        nSeries = np.asarray(self.Series[sMetricName]).astype(np.float32)
        self.Values.append(nSeries)
        if oMetric.Reduction.upper() == "NONE":
          self.__currentMetrics.append(nSeries)
        if oMetric.Reduction.upper() == "AVG":
          self.__currentMetrics.append(nSeries.mean())
        elif oMetric.Reduction.upper() == "SUM":
          self.__currentMetrics.append(np.sum(nSeries))
        elif oMetric.Reduction.upper() == "SQRT":
          self.__currentMetrics.append(np.sqrt(nSeries.mean()))
    
    if p_bIsClearing:
      self.Clear()
    
    self.Current = CMetricsPacket([x for x in self.keys()], self.__currentMetrics)
    

    return self
  #--------------------------------------------------------------------------------------------------------------
  def Unpack(self):
    if len(self.__currentMetrics) == 1:
      return oMetrics[0]
    else:
      return tuple(self.__currentMetrics) 
  #--------------------------------------------------------------------------------------------------------------
  def getMetricName(self, p_sMetricName, p_bIsUpperCase=True):
    sMetricFullName = p_sMetricName
    if self.MetricNamePrefix is not None:
      sMetricFullName = self.MetricNamePrefix + "_" + sMetricFullName
      
    if p_bIsUpperCase:
      sMetricFullName = sMetricFullName.upper()
    return sMetricFullName
  #--------------------------------------------------------------------------------------------------------------    
  def AsReportRow(self, p_sFormatStr=None, p_bIsClearing=True, p_bIsUpperCase=True):
    sFormat = p_sFormatStr
    #oValues = list(self.Aggregate(p_bIsClearing=p_bIsClearing))
    if sFormat is None:
      sResult = ""  
      for nIndex, sMetricName in enumerate(self):
        sMetricFullName = self.getMetricName(sMetricName, p_bIsUpperCase)
        sResult += f"| {sMetricFullName}:{self.__currentMetrics[nIndex]:.4f} "
    else:
      oArgs = []
      for nIndex, sMetricName in enumerate(self):
        oArgs.append(self.getMetricName(sMetricName, p_bIsUpperCase))
        oArgs.append(self.__currentMetrics[nIndex])
      sResult = sFormat % tuple(oArgs)
      
    sResult += "| Elapsed:%.1fs" % self.PeriodDuration.total_seconds()
    return sResult
  #--------------------------------------------------------------------------------------------------------------    
  def AsReportPage(self, p_bIsClearing=True, p_bIsUpperCase=True):
    #oValues = list(self.Aggregate(p_bIsClearing=p_bIsClearing))
    nMaxNameLen = 0
    for sMetricName in self:
      sMetricFullName = self.getMetricName(sMetricName, p_bIsUpperCase)
      if len(sMetricFullName) > nMaxNameLen:
        nMaxNameLen = len(sMetricFullName)

    sResult = " Metrics:\n"  
    for nIndex, sMetricName in enumerate(self):
      sMetricFullName = self.getMetricName(sMetricName, p_bIsUpperCase)
      sResult += f"  |__ {sMetricFullName.ljust(nMaxNameLen)}:{self.__currentMetrics[nIndex]:.4f}\n"
      
    return sResult    
  #--------------------------------------------------------------------------------------------------------------    
  def AsCSV(self, p_sDelimiter=";", p_bIsClearing=True, p_bIsUpperCase=True):
    #oValues = list(self.Aggregate(p_bIsClearing=p_bIsClearing))
    sHeaders = ""
    sValues  = ""
    for nIndex, sMetricName in enumerate(self):
      sMetricFullName = self.getMetricName(sMetricName, p_bIsUpperCase)
      sHeaders += sMetricFullName + p_sDelimiter
      sValues  += f"{self.__currentMetrics[nIndex]:.4f}{p_sDelimiter}" 
    sHeaders = sHeaders[:-1]
    sValues  = sValues[:-1]
    sResult  = sHeaders + "\n" + sValues
    
    return sResult    
  #--------------------------------------------------------------------------------------------------------------    
#==============================================================================================================================    
    
if __name__ == "__main__":
  
  def calcMetrics(p_nPrediction, p_nTarget):
    d = np.subtract(p_nPrediction, p_nTarget)
    mae = np.abs(d)
    mse = d**2
    
    return mae, mse
    
  oMetrics = CMetricsCollection("val")
  oMetrics.Register(["loss", "loss2"])
  oMetrics.RegisterFunction(["mae", "mse"], calcMetrics)
  oMetrics["mse"].Reduction = "sqrt"
  
  np.random.seed(2023)
  nPred   = np.random.rand(100, 8, 10)
  nTarget = np.random.rand(100, 8, 10)
  nBatchSize = 10
  for nIndex in range(100//nBatchSize):
    oMetrics.Calculate( nTarget[nIndex*nBatchSize:(nIndex+1)*nBatchSize, :]
                       , nPred[nIndex*nBatchSize:(nIndex+1)*nBatchSize, :]) 
    l = np.random.rand(1)
    oMetrics.Collect("loss", l)
    oMetrics.Collect("loss2", l + 1)
  
  
  
  print(oMetrics.AsReportRow(p_bIsClearing=False, p_bIsUpperCase=False))
  
  print(oMetrics.AsReportRow("%s: %.4f  %s:%.4f %s:%.4f %s:%.4f",
                p_bIsClearing=False))
  print("")
  print(oMetrics.AsReportPage(p_bIsClearing=False))
  a, b, c, d = oMetrics.Aggregate()
  print(a, b, c, d)
  oMetrics.Restore()
  a, b, c, d = oMetrics.Aggregate()
  print(a, b, c, d)
  if oMetrics.Values[0] is not None:
    print("Shape of metrics", oMetrics.Values[1].shape)  
  
  
      