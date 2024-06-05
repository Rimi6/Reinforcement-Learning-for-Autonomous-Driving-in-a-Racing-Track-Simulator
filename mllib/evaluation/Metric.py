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


#==============================================================================================================================
class CMetric(object):
  #--------------------------------------------------------------------------------------------------------------
  def __init__(self, p_sName, p_fBatchFunction=None, p_sReduction="avg"):
    self.Name = p_sName
    if p_fBatchFunction is None: 
      self.Function = self
    else:
      self.Function   = p_fBatchFunction
    self.Reduction    = p_sReduction
    self.Index        = 0
    self.Meta         = dict()
  #--------------------------------------------------------------------------------------------------------------
  @property
  def FunctionName(self):
    if self.Function == self:
      return ":" + str(type(self))
    else:
      return self.Function.__name__ + "()"
  #--------------------------------------------------------------------------------------------------------------      
  def __call__(self, *args):
    return None
  #--------------------------------------------------------------------------------------------------------------
  def __str__(self)->str:
    sResult = self.Name + " " + self.FunctionName
    return sResult
  #--------------------------------------------------------------------------------------------------------------
  def __repr__(self)->str:
    return self.__str__()
  #--------------------------------------------------------------------------------------------------------------
#==============================================================================================================================   