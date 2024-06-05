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


import numpy as np

# --------------------------------------------------------------------------------------
def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray
# --------------------------------------------------------------------------------------




# =========================================================================================================================
class Box2DWrapper():
    """
    Environment wrapper for CarRacing 
    """
    IS_GYMNASIUM = True
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfig, env, p_bIsRendering=False):

        self.Config = p_oConfig
        
        self.image_stack_count = self.Config["Data.ImageStackCount"]
        
        self.env = env  
        self.IsRendering = p_bIsRendering
        self.action_repeat = self.Config["Data.ActionRepeatCount"]
    # --------------------------------------------------------------------------------------
    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        if Box2DWrapper.IS_GYMNASIUM:
          img_rgb, _ = self.env.reset()
        else:
          img_rgb = self.env.reset()
          
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * self.image_stack_count  # four frames for decision
        return np.array(self.stack)
    # --------------------------------------------------------------------------------------
    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            if self.IsRendering:
                self.env.render()
                
            oResult = self.env.step(action)
            if Box2DWrapper.IS_GYMNASIUM:
              img_rgb, reward, die, truncated, info = oResult
            else:
              truncated = False
              img_rgb, reward, die, info = oResult
              
               
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.image_stack_count
        return np.array(self.stack), total_reward, done, die
    # --------------------------------------------------------------------------------------
    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
    # --------------------------------------------------------------------------------------
# =========================================================================================================================    