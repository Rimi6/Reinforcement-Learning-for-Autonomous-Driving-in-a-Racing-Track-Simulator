# ......................................................................................
# MIT License

# Copyright (c) 2023 Pantelis I. Kaplanoglou

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


import av
import numpy as np

from PIL import Image, ImageOps

# =========================================================================================================================
class VideoRecorderAV(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, filename, resolution):
    self.filename = filename
    self.width, self.height = resolution
  # --------------------------------------------------------------------------------------
  def start(self):
    # Create a PyAV output container and a video stream with the desired properties
    self.container = av.open(self.filename, 'w')
    self.video_stream = self.container.add_stream('libx264', rate=25)  # Adjust the codec and frame rate as needed
    self.video_stream.width  = self.width
    self.video_stream.height = self.height
  # --------------------------------------------------------------------------------------  
  def add_frame(self, n_image):
    oImage = Image.fromarray(n_image)    
    oImage = ImageOps.flip(oImage)
    #oImage = oImage.resize((nImage.shape[1]//2, nImage.shape[0]//2))
                
    # Convert the image data to RGB format
    oFrame = av.VideoFrame.from_ndarray(np.array(oImage), format='rgb24')
    
    # Encode and write the frame to the video stream
    packet = self.video_stream.encode(oFrame)
    self.container.mux(packet)
  # --------------------------------------------------------------------------------------      
  def end(self):      
    for packet in self.video_stream.encode():
      self.container.mux(packet)
    self.container.close()    
  # --------------------------------------------------------------------------------------
# =========================================================================================================================        