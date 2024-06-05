import time
import numpy as np
from mllib import CFileStore
from models.baseline import PurePursuitPlanner
from envs import F1RaceTrack, VideoRecorderAV
import matplotlib.pyplot as plt
#import os
#os.environ['PYGLET_SHADOW_WINDOW'] = '1'

def main():
  IS_FAST_RENDERING = True
  IS_RECORDING      = True
  if IS_RECORDING:
    IS_FAST_RENDERING = True
  IS_VISUALIZING    = True
  IS_PREVIEW        = True
  
  SELECTED_RACETRACK = "example"
  #SELECTED_RACETRACK = "Catalunya"
  #SELECTED_RACETRACK = "Austin"
  #SELECTED_RACETRACK = "Monza"

  SPEED_TO_REAL_RATIO = 1
  SPEED_LIMIT = 40
  MAX_BRAKING = -np.ceil(SPEED_LIMIT/1.8)
  print(f"Speed Limit {SPEED_LIMIT:.0f} Max Braking {MAX_BRAKING:.0f}")
  
  VEHICLE_CONFIG = {   "tlad" : { "value": 0.82461887897713965 ,"descr": "Lookahead distance for the pure pursuit planner" }
                      ,"vgain": { "value": 1.375               ,"descr": "Deprecated, it is calcualted by Agent.SpeedLimitKPH" }
                      ,"Agent.SpeedLimitKPH"  : { "value": SPEED_LIMIT  ,"descr": "The maximum speed allowed for the self-driving agent" }
                      ,"Agent.MaxBrakingInKPH": { "value": MAX_BRAKING  ,"descr": "The maximum negative speed (for braking) that can be set" }
                    }
  
    
  
  
    
  '''
  # Vehicle Physics
  mu         : surface friction coefficient
  C_Sf       : Cornering stiffness coefficient, front
  C_Sr       : Cornering stiffness coefficient, rear
  lf         : Distance from center of gravity to front axle
  lr         : Distance from center of gravity to rear axle
  h          : Height of center of gravity
  m          : Total mass of the vehicle
  I          : Moment of inertial of the entire vehicle about the z axis
  s_min      : Minimum steering angle constraint
  s_max      :   Maximum steering angle constraint
  sv_min     : Minimum steering velocity constraint
  sv_max     : Maximum steering velocity constraint
  v_switch   : Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
  a_max      : Maximum longitudinal acceleration
  v_min      : Minimum longitudinal velocity
  v_max      : Maximum longitudinal velocity
  width      : width of the vehicle in meters
  length     : length of the vehicle in meters
  '''            
          
  dParams = {     'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562
                , 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74
                , 'I': 0.04712
                , 's_min': -0.4189, 's_max': 0.4189
                , 'sv_min': -3.2, 'sv_max': 3.2
                , 'v_switch': 7.319, 'a_max': 9.51
                , 'v_min': VEHICLE_CONFIG["Agent.MaxBrakingInKPH"]["value"] / 3.6
                , 'v_max': VEHICLE_CONFIG["Agent.SpeedLimitKPH"]["value"] / 3.6
                , 'width': 0.31, 'length': 0.58
                , 'speed_adjust_to_real': SPEED_TO_REAL_RATIO
                } 
    
  # Create file stores and determine filenames
  oVideosFS  = CFileStore("Videos")
  if SELECTED_RACETRACK == "example":
    sVideoFilename = oVideosFS.File("race_baseline.mp4")
  else:
    sVideoFilename = oVideosFS.File(SELECTED_RACETRACK + ".mp4")
  
  oRacetracksFS = CFileStore("f1tenth_racetracks") 
  print("-"*40, "Videos", "-"*40)
  print(oVideosFS.DirectoryEntries)

  
  # Create the racetrack 
  oRaceTrack = F1RaceTrack(oRacetracksFS.File(SELECTED_RACETRACK), 
                            is_fast_rendering=IS_FAST_RENDERING, is_recording=IS_RECORDING, car_params=dParams) 
    

  if IS_RECORDING:
    oRecorder  = VideoRecorderAV(sVideoFilename, (1000,800))
    oRecorder.start()
    print(f"Recording video: {sVideoFilename}")  
  
  oPlanner = PurePursuitPlanner(oRaceTrack.config, (0.17145+0.15875), oRaceTrack.raceline
                               , lookahead_dist=VEHICLE_CONFIG['tlad']["value"]
                               , speed_limiter=VEHICLE_CONFIG["Agent.SpeedLimitKPH"]["value"]) 
  #planner = FlippyPlanner(speed=0.2, flip_every=1, steer=10)
  #planner = BabyDriver()

  
  # Initializes the environment and shows a screenshot
  state, step_interval_secs, done, info = oRaceTrack.reset(pure_pursuit_planner=oPlanner)
  if IS_VISUALIZING or IS_PREVIEW:
    nImage = oRaceTrack.render()

    plt.imshow(oRaceTrack.screenshot)
    plt.show()

  # Runs a complete episode (2 laps)
  two_laps_time = 0.0
  dtStartSimulation = time.time()
  
  oSpeeds = []
  while not done:
    # Agent decides action based on data from the current state
    vehicle_x, vehicle_y, vehicle_theta = state['poses_x'][0], state['poses_y'][0], state['poses_theta'][0]
    speed, steer = oPlanner.plan(vehicle_x, vehicle_y, vehicle_theta)
    
    # Agent performs action on the environment and it transitions to a new state (effects of the action)
    nActionOnEnv = np.array([[steer, speed]])                            
    state, step_interval_secs, done, info = oRaceTrack.step(nActionOnEnv)
    
    nSpeed = oRaceTrack.vehicle_speed(state)
    oSpeeds.append(nSpeed)
    
    two_laps_time += step_interval_secs

    if IS_VISUALIZING:    
      nImage = oRaceTrack.render()
      if IS_RECORDING and oRaceTrack.env.is_period_start:
        oRecorder.add_frame(nImage)
  
  nSpeeds = np.array(oSpeeds)
  
  nElapsed = time.time() - dtStartSimulation
  
  
  nLapCount,nRaceTime = state["lap_counts"][0], state["lap_times"][0]
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
      
      
      
  if IS_RECORDING:
    oRecorder.end()

if __name__ == '__main__':
    main()
