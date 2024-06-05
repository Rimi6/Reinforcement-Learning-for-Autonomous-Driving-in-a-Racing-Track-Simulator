#import gymnasium as gym
#from gymnasium import logger as gymlogger
#from gymnasium.wrappers import RecordVideo

from envs import Box2DWrapper


if Box2DWrapper.IS_GYMNASIUM:
  import gymnasium as gym
  from gymnasium import logger as gymlogger
else:
  import gym
  from gym import logger as gymlogger
#from gym.wrappers import RecordVideo

gymlogger.set_level(40) #error only

#import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

import time
import math
import glob
import io
import base64

import torch
from collections import deque


from mllib import CFileStore
from mllib.experiment import CMLExperimentConfig
from models import Box2DAgent


oConfigFS = CFileStore("MLModels")

CONFIG = CMLExperimentConfig(oConfigFS.File("RLExperiment8.json"))

MODEL_NUMBER = CONFIG["ModelNumber"]
sModelFolder = "%s_%.2d" % (CONFIG["ModelName"], CONFIG["ModelNumber"])
oModelFileStore = CFileStore("MLModels").SubFS(sModelFolder)

RANDOM_SEED = CONFIG["RandomSeed"]




# Initializing Training Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


if Box2DWrapper.IS_GYMNASIUM:
  # Gymnasium 0.28
  env = gym.make('CarRacing-v2', verbose=0, render_mode='rgb_array')
else:
  # Gym 0.17
  env = gym.make('CarRacing-v0', verbose=0)

state = env.reset()
print('env.action_space.shape: ', env.action_space.shape)
reward_threshold = env.spec.reward_threshold
print('reward_threshold', reward_threshold)


def save(agent, filename):
    torch.save(agent.net.state_dict(), filename)


def load(agent, filename):
    agent.net.load_state_dict(torch.load(filename))  
    

            
            




IS_RETRAINING = False

oAgent = Box2DAgent(device, CONFIG)
oEnvWrapper = Box2DWrapper(CONFIG,env)



sModelStateFileName = "model%d_weights.pth" % MODEL_NUMBER


bMustTrain = (not oModelFileStore.Exists(sModelStateFileName)) or IS_RETRAINING


if bMustTrain:
  print("[>] Training")
  #scores, avg_scores  = ppo_train(oEnvWrapper, oAgent, n_episodes=NUM_EPISODES, save_every=100)
  SAVE_EVERY = 100
  

  scores_deque = deque(maxlen=100)
  scores_array = []
  avg_scores_array = []    

  timestep_after_last_save = 0
  
  time_start = time.time()

  running_score = 0
  state = oEnvWrapper.reset()
  
  i_lim = 0
  
  NUM_EPISODES  = CONFIG["Training.RL.NumEpisodes"]
  # Duration of Episode ~20secs
  #print("Estimated training time %d mins" % ((NUM_EPISODES*20.0)/0.0))
  print("Estimated training time %d hours" % ((NUM_EPISODES*8.0)/2000.0))  
  
  for i_episode in range(NUM_EPISODES):
      
      timestep = 0
      total_reward = 0
      
      ## score = 0
      state = oEnvWrapper.reset()

      while True:    
          action, a_logp = oAgent.act(state)
          
          action_vector = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
          next_state, reward, done, die  = oEnvWrapper.step(action_vector)
          
          oAgent.perceive(reward, next_state)
          
          total_reward += reward
          state = next_state
          
          timestep += 1  
          timestep_after_last_save += 1
          
          if done or die:
              break
              
      running_score = running_score * 0.99 + total_reward * 0.01

      scores_deque.append(total_reward)
      scores_array.append(total_reward)

      avg_score = np.mean(scores_deque)
      avg_scores_array.append(avg_score)
      
      s = (int)(time.time() - time_start)        
      print('Ep. {}, Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Run.Score {:.2f}, \
Time: {:02}:{:02}:{:02} '\
          .format(i_episode, timestep, \
                  total_reward, avg_score, running_score, s//3600, s%3600//60, s%60))  
     
      
      # Save episode is equal to "save_every" timesteps
      if i_episode+1 % SAVE_EVERY == 0:

          suf = str(i_episode)
          save(oAgent, '', 'model_weights', suf)
          
      if np.mean(scores_deque) > reward_threshold:
          print("Solved environment! Running score is {:.2f}, Avg.Score: {:.2f} !" \
                .format(running_score, avg_score))
          break
          
            
  oModelFileStore.Serialize("Scores.pkl", scores_array)
  oModelFileStore.Serialize("AverageScores.pkl", avg_scores_array)

  # Save latest model. We'll use it for testing
  save(oAgent, oModelFileStore.File(sModelStateFileName))
  
else:
  print("[>] Loading")
  scores = oModelFileStore.Deserialize("Scores.pkl")
  avg_scores = oModelFileStore.Deserialize("AverageScores.pkl")
  load(oAgent, oModelFileStore.File(sModelStateFileName))             


