# MIT License

# Copyright (c) 2021 Eoin Gogarty, Charlie Maguire and Manus McAuliffe (Formula Trintiy Autonomous)

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

"""
Stable Baselines 3 training script for F1Tenth Gym with vectorised environments
"""

import os
import gymnasium as gym
import time
import glob
import wandb
import argparse
import numpy as np

from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from custom.eoin_callbacks import SaveOnBestTrainingRewardCallback

from envs import F1RaceTrack, VideoRecorderAV
from envs.rewards import F1RaceBasicReward, F1RaceReward

from mllib import CFileStore, CConfigArgs

def main(args, p_oConfig):
    sModelFolder = "%s_%.2d" % (p_oConfig["ModelName"], p_oConfig["ModelNumber"])
    oModelFS = CFileStore("MLModels").SubFS(sModelFolder)
    oCheckpointFS = oModelFS.SubFS("checkpoints")
    oLogFS = oModelFS.SubFS("tboard")
    oVideosFS = CFileStore("Videos")
    oRacetracksFS = CFileStore("f1tenth_racetracks") 
    
    RANDOM_SEED = p_oConfig["RandomSeed"]
    #IS_RECORDING = p_oConfig["IsRecordingVideo"]
    #       #
    # TRAIN #
    #       #

    # initialise weights and biases
    if args.wandb:
        wandb.init(sync_tensorboard=True)
        
    SPEED_LIMIT = p_oConfig["Agent.SpeedLimitKPH"]
    MAX_BRAKING = p_oConfig["Agent.MaxBrakingInKPH"]
    print(f"Speed Limit {SPEED_LIMIT:.0f} Max Braking {MAX_BRAKING:.0f}")
      
    nSpeedLimit  = SPEED_LIMIT / 3.6 # m/s
    nMaxBreaking = MAX_BRAKING / 3.6 # m/s

    SPEED_TO_REAL_RATIO = 1
    dParams = {  'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562
                  , 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74
                  , 'I': 0.04712
                  , 's_min': -0.4189, 's_max': 0.4189
                  , 'sv_min': -3.2, 'sv_max': 3.2
                  , 'v_switch': 7.319, 'a_max': 9.51
                  , 'v_min':nMaxBreaking, 'v_max': nSpeedLimit
                  , 'width': 0.31, 'length': 0.58
                  , 'speed_adjust_to_real': SPEED_TO_REAL_RATIO
                }          
                         
    SELECTED_RACETRACK = p_oConfig["Environment.TrackName"]
                    
    oRaceTrack    = F1RaceTrack(oRacetracksFS.File(SELECTED_RACETRACK), is_fast_rendering=False, is_recording=True, car_params=dParams)
    oF1RaceReward = F1RaceReward(oRaceTrack, p_oConfig)
        
    # prepare the environment
    def wrap_env():
      return oF1RaceReward

    '''
    def wrap_env():
        # starts F110 gym
        env = gym.make("f110_gym:f110-v0",
                       map=MAP_PATH,
                       map_ext=MAP_EXTENSION,
                       num_agents=1)
        # wrap basic gym with RL functions
        env = F110_Wrapped(env)
        env = RandomMap(env, MAP_CHANGE_INTERVAL)
        return env
    '''
    start_time = time.time()
    
    # create log directory for monitor wrapper (in make_vec_env)
    # vectorise environment (parallelise)
    envs = make_vec_env(wrap_env,
                        n_envs=p_oConfig["Training.ParallelProcessCount"],
                        seed=2023,
                        monitor_dir=oLogFS.BaseFolder,
                        vec_env_cls=SubprocVecEnv)

    # load or create model
    model, reset_num_timesteps = load_model(args.load,
                                            oCheckpointFS.BaseFolder,
                                            envs,
                                            oLogFS.BaseFolder)
    print("[>] Loaded saved RL model")
    env_eval = wrap_env()

    if args.IsRecordingVideo:
      oRecorder  = VideoRecorderAV(oVideosFS.File("race_%.2d.mp4" % p_oConfig["ModelNumber"]), (1000,800))
      oRecorder.start()  
      
    dtStartSimulation = time.time()
    
    obs = env_eval.reset()
    env_eval.render()
    
    nSteps = 0
    oSpeeds = []
    done = False
    while not done:
        # use trained model to predict some action, using observations
        action, _ = model.predict(obs)
        obs, _, done, _ = env_eval.step(action)
    
        state, _, _, _ = oRaceTrack.last_state
        nSpeed = oRaceTrack.vehicle_speed(state)
        oSpeeds.append(nSpeed)
    
        nImage = env_eval.render()
        if oRaceTrack.env.is_period_start:
          oRecorder.add_frame(nImage)
    
        if nSteps % 100 == 0:
          print(f"Step {nSteps} Speed: {nSpeed:.1f} Km/h")      
        nSteps+= 1
                
    nSpeeds = np.array(oSpeeds)
    nElapsed = time.time() - dtStartSimulation
    
    if args.IsRecordingVideo:
      oRecorder.end()  

    state, _, _, _ = oRaceTrack.last_state
    nLapCount,nRaceTime = state["lap_counts"][0], state["lap_times"][0]
    vehicle_theta = state["poses_theta"][0]
    if nLapCount > 0:
      nLapTime = nRaceTime / nLapCount
      
      print(f"Race results:")
      print(f"  |__ Vehicle max speed: {SPEED_LIMIT*SPEED_TO_REAL_RATIO} Km/h")
      print(f"  |__ Average Lap Time:{nLapTime:.3f}sec | Average Speed:{nSpeeds.mean():.1f} Km/h")
      print(f"  |__ Race time:{nRaceTime:.3f}sec (simulation run time: {nElapsed:.3f}sec)")
    else:
      bHasCrashed = state['collisions'][0]
      bHasSpun    = abs(vehicle_theta) > 100
      if bHasCrashed:
        print("Vehicle has crashed!")
        print(f"  |__ At race time:{nRaceTime:.3f}sec (simulation run time: {nElapsed:.3f}sec)")
        print(f"  |__ Turn Angle: {vehicle_theta:.3f}")
        print(f"  |__ Vehicle max speed: {SPEED_LIMIT*SPEED_TO_REAL_RATIO} Km/h")
      elif bHasSpun:
        print("Vehicle has spun!")
        print(f"  |__ At race time:{nRaceTime:.3f}sec (simulation run time: {nElapsed:.3f}sec)")
        print(f"  |__ Turn Angle: {vehicle_theta:.3f}")
        print(f"  |__ Vehicle max speed: {SPEED_LIMIT*SPEED_TO_REAL_RATIO} Km/h")  




def load_model(load_arg, train_directory, envs, tensorboard_path=None, evaluating=True):
    '''
    Slighly convoluted function that either creates a new model as specified below
    in the "create new model" section, or loads in the latest trained
    model (or user specified model) to continue training
    '''
    reset_num_timesteps = False
    # get trained model list
    trained_models = glob.glob(f"{train_directory}/*")
    # latest model
    if (load_arg == "latest") or (load_arg is None):
        model_path = max(trained_models, key=os.path.getctime)
    else:
        trained_models_sorted = sorted(trained_models,
                                       key=os.path.getctime,
                                       reverse=True)
        # match user input to model names
        model_path = [m for m in trained_models_sorted if load_arg in m]
        model_path = model_path[0]
    # get plain model name for printing
    model_name = model_path.replace(".zip", '')
    model_name = model_name.replace(f"{train_directory}/", '')
    print(f"Loading model ({train_directory}) {model_name}")
    # load model from path
    model = PPO.load(model_path)
    # set and reset environment
    model.set_env(envs)
    envs.reset()
    # return new/loaded model
    return model, reset_num_timesteps


# necessary for Python multi-processing
if __name__ == "__main__":
    # parse runtime arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-l",
                        "--load",
                        help="load previous model",
                        nargs="?",
                        const="latest")
    parser.add_argument("-w",
                        "--wandb",
                        help="use Weights and Biases API",
                        action="store_true")
    parser.add_argument("-s",
                        "--save",
                        help="always save at step interval",
                        action="store_true")
    args = parser.parse_args()
    # call main training function
    
    MODEL_FOLDER = "MLModels"
    EXPERIMENT_CODE = "RLF20"
    # Creates a filestore object and uses its method to load the config dict from a JSON file 
    oConfigFS = CFileStore(MODEL_FOLDER)
    CONFIG = oConfigFS.LoadJSON(EXPERIMENT_CODE + ".json", p_sErrorTemplate="Hyperparameter configuration file %s not found.")
    # Deserializes the args object from the config dict. That dynamically adjs fields to the args objects
    args = CConfigArgs(CONFIG)
    
    main(args, CONFIG)
