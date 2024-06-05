# MIT License

# Copyright (c) 2020 FT Autonomous Team One

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

import gym
import numpy as np

from gym import spaces
from pathlib import Path

from custom.random_trackgen import create_track, convert_track

# --------------------------------------------------------------------------------------
def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min
# --------------------------------------------------------------------------------------    




# =========================================================================================================================
class F1RaceReward(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios
    """
    # --------------------------------------------------------------------------------------
    def __init__(self, env, hyperparams):
        super().__init__(env)
        
        self.hyperparams = hyperparams
        
        # normalised action space, steer and speed
        self.action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64)

        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1080,), dtype=np.float64)

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

        # store car dimensions and some track info
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 3.2  # ~= track width, see random_trackgen.py

        # radius of circle where car can start on track, relative to a centerpoint
        self.start_radius = (self.track_width / 2) - \
            ((self.car_length + self.car_width) / 2)  # just extra wiggle room

        self.elapsed_moments = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100
        self.next_waypoint_index = 0
        self.total_reward = 0
        self.config = self.env.config
        self.waypoints = self.env.waypoints
        self.max_waypoint_index = len(self.waypoints) - 1
        self.total_run_time = 0
        self.passed_waypoint_count = 0
        self.interval_count = 0
        self.logger = None
        #self.stuck_steps      = self.hyperparams["Agent.StuckSteps"]
        #self.slow_mover_steps = self.hyperparams["Agent.SlowMoverSteps"]
    # --------------------------------------------------------------------------------------
    def is_on_line_segment_between(self, x, y, wp_before_index, wp_after_index):
      x1, y1  = self.waypoints[wp_before_index]
      x2, y2  = self.waypoints[wp_after_index]
                  
      cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
      if cross_product != 0:
          return False

      dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
      if dot_product < 0:
          return False

      squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
      if dot_product > squared_length:
          return False

      return True        
    # --------------------------------------------------------------------------------------
    def calc_distances(self, x, y, prev_wp_index, next_wp_index, future_wp_index):
      nPrev_x, nPrev_y  = self.waypoints[prev_wp_index]
      nNext_x, nNext_y  = self.waypoints[next_wp_index]
      nFutu_x, nFutu_y  = self.waypoints[future_wp_index]
      
      nDistPrev = np.sqrt(np.power((x - nPrev_x), 2) + np.power((y - nPrev_y), 2))
      nDistNext = np.sqrt(np.power((x - nNext_x), 2) + np.power((y - nNext_y), 2))
      nDistFutu = np.sqrt(np.power((x - nFutu_x), 2) + np.power((y - nFutu_y), 2))
      return nDistPrev, nDistNext, nDistFutu
    # --------------------------------------------------------------------------------------
    def determine_car_progress(self, x, y):
        nNextIndex =  self.next_waypoint_index
        if nNextIndex == 0:
          nPrevIndex = self.max_waypoint_index
        else:
          nPrevIndex = nNextIndex - 1
        
        if nNextIndex < self.max_waypoint_index:
          nFutureIndex = nNextIndex + 1
        else:
          nFutureIndex = 0            
          
        nDistPrev, nDistNext, nDistFuture = self.calc_distances(x,y, nPrevIndex, nNextIndex, nFutureIndex)  
        
        # Move closer to the next waypoint in comparison to its previous
        nClosestIndex = nPrevIndex 
        if (nDistNext < nDistPrev) or (np.abs(nDistNext - nDistPrev) < 1e-6):
            if (nDistNext > nDistFuture) or (np.abs(nDistNext - nDistFuture) < 1e-6):
                # It has passed the waypoint
                nClosestIndex = nFutureIndex
            else: 
                nClosestIndex = nNextIndex
                
        #print(f"{nDistPrev:.4f}->{nDistNext:.4f}->{nDistFuture:.4f} {nPrevIndex},{nNextIndex},{nFutureIndex} Closest:{nClosestIndex}")         
        bMoved = (nClosestIndex == nFutureIndex)
        
        return bMoved, nClosestIndex, nPrevIndex, nNextIndex, nFutureIndex    
    # --------------------------------------------------------------------------------------
    def step(self, action):
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        observation, run_time , done, info = self.env.step(np.array([action_convert]))
        self.total_run_time += run_time
        
        x,y,theta = observation['poses_x'][0], observation['poses_y'][0], observation['poses_theta'][0]
        
        self.elapsed_moments += 1

        # TODO -> do some reward engineering here and mess around with this
        reward = 0
        
        nWaypoints = len(self.waypoints)
        complete_pc = self.passed_waypoint_count/nWaypoints
        
        bHasCrashed       = observation['collisions'][0] > 0
        bHasSpun          = abs(theta) > self.max_theta
        bIsCheckeredFlag  = False
        
        # end episode if car is spinning
        if bHasCrashed:
            if self.total_reward > 0.0:
              reward = -self.total_reward*(1-complete_pc)
            else:
              reward = -10.0
            print(f"Crash {self.passed_waypoint_count}/{nWaypoints} wp#{self.next_waypoint_index} tot.reward: {self.total_reward} penalty:{reward}", flush=True)
            done = True
        elif bHasSpun:
            if self.total_reward > 0.0:
              reward = -self.total_reward#*(1-complete_pc)
            else:
              reward = -10.0
            print(f"Spin {self.passed_waypoint_count}/{nWaypoints} wp#{self.next_waypoint_index} tot.reward: {self.total_reward} penalty:{reward}", flush=True)
            done = True
        elif done:
          #bIsCheckeredFlag = True
          print(f"Stopped {self.passed_waypoint_count}/{nWaypoints} wp#{self.next_waypoint_index} tot.reward: {self.total_reward} penalty:{reward}", flush=True)
          nLapCount,nRaceTime = observation["lap_counts"][0], observation["lap_times"][0]
          if nLapCount > 0:
            nLapTime = nRaceTime / nLapCount
              
          print(f" |_ Laps done:{nLapCount} average lap time:{nLapTime:.3f} extra reward:{reward}", flush=True)
          
        ''' 
        if bIsCheckeredFlag:
          nLapsCompleted = observation["lap_counts"]
          reward = self.total_reward*(nWaypoints/self.elapsed_moments)
          self.total_reward += reward
          print(f" |_ Laps done {nLapsCompleted} extra reward:{reward}")
        '''
          
        '''
        if (not done): 
          if self.passed_waypoint_count <= 10:
            nLimit = self.stuck_steps
            if (self.interval_count >= nLimit):
              if self.total_reward > 0.0:
                reward - self.total_reward
              else:
                reward = -10.0
              done = True            
          else:
            nLimit = self.slow_mover_steps*self.passed_waypoint_count
            if (self.interval_count >= nLimit):
              #if self.total_reward > 0.0:
              #reward = -self.total_reward*(1.0-np.exp(-complete_pc/2.0))
              print(f"Stuck {self.passed_waypoint_count}/{nWaypoints} wp#{self.next_waypoint_index} tot.reward: {self.total_reward} penalty:{reward}")
              done = True
        '''
          
        if not done:
            #eoins reward function
            vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
            #[PANTELIS] reward = vel_magnitude #/10 maybe include if speed is having too much of an effect
    
            # reward function that returns percent of lap left, maybe try incorporate speed into the reward too
            if self.next_waypoint_index is None:
                nIndex, nCoords, nDist = self.env.get_nearest_waypoint(x,y)
                self.next_waypoint_index = nIndex
            
            # TODO: Perhaps this can be replaced with a call to get_nearest_waypoint 
            bMoved, nClosestIndex, nPrevIndex, nNextIndex, nFutureIndex  = self.determine_car_progress(x, y)  
            #print(f"{x:.4f},{y:.4f} closest wp:{nClosestIndex} act:{action} vel:{observation['linear_vels_x'][0]:.4f},{observation['linear_vels_y'][0]:.4f} {nPrevIndex}->{nNextIndex}->{nFutureIndex}")
            

            if bMoved:        
                #speed_ratio = ((self.passed_waypoint_count + 1)*1000)/self.time_index
                reward =  np.exp(2.0*complete_pc)# * speed_ratio
               
                self.passed_waypoint_count += 1
                self.next_waypoint_index += 1
                #self.interval_count = 0
                
                if self.next_waypoint_index > self.max_waypoint_index:
                  self.next_waypoint_index = 0  
            else:
                self.interval_count += 1
              
            '''
            if (not bMoved) and (self.interval_count >= self.slow_mover_steps):
              if self.passed_waypoint_count > 0:
                reward = -(1.0 / self.passed_waypoint_count)*10.0
              else:
                reward = -1.0/(self.stuck_steps-self.slow_mover_steps)
            '''         
            
            bIsCheckeredFlag = self.passed_waypoint_count >= nWaypoints
            if bIsCheckeredFlag:
              reward += self.total_reward*(1000.0/self.elapsed_moments)
              print(f"Lap completed reward {reward:.1f}", self.next_waypoint_index, self.total_reward, flush=True)
              done=True

            self.total_reward += reward
        else:
          self.reset_car()
          
          
        return self.normalise_observations(observation['scans'][0]), reward, bool(done), info
        
        '''
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        #print("V:",vel_magnitude)
        if vel_magnitude > 0.2:  # > 0 is very slow and safe, sometimes just stops in its tracks at corners
            reward += 0.1"""

        # penalise changes in car angular orientation (reward smoothness)
        """ang_magnitude = abs(observation['ang_vels_z'][0])
        #print("Ang:",ang_magnitude)
        if ang_magnitude > 0.75:
            reward += -ang_magnitude/10
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 5:
            reward = -(vel_magnitude/10)
        '''
    # --------------------------------------------------------------------------------------   
    def reset_car(self):
        self.passed_waypoint_count = 0
        self.next_waypoint_index   = None
        self.total_reward          = 0
        self.elapsed_moments            = 0
        self.interval_count        = 0
    # --------------------------------------------------------------------------------------        
    def reset(self, start_xy=None, direction=None, initial_rotation=None):
        observation, _, _, _ = self.env.reset(self.env.init_vector)

        self.reset_car()
        # reward, done, info can't be included in the Gym format
        return self.normalise_observations(observation['scans'][0])
    # --------------------------------------------------------------------------------------
    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float64)
    # --------------------------------------------------------------------------------------
    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])
    # --------------------------------------------------------------------------------------
    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None
    # --------------------------------------------------------------------------------------
    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")
    # --------------------------------------------------------------------------------------
    def render(self):
        return self.env.render()
    # --------------------------------------------------------------------------------------
# =========================================================================================================================



    
    
    
    
    
    
    
# =========================================================================================================================          
class RandomMap(gym.Wrapper):
    """
    Generates random maps at chosen intervals, when resetting car,
    and positions car at random point around new track
    """

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20
    # --------------------------------------------------------------------------------------
    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0
    # --------------------------------------------------------------------------------------
    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # create map
            for _ in range(self.MAX_CREATE_ATTEMPTS):
                try:
                    track, track_int, track_ext = create_track()
                    convert_track(track,
                                  track_int,
                                  track_ext,
                                  self.current_seed)
                    break
                except Exception:
                    print(
                        f"Random generator [{self.current_seed}] failed, trying again...")
            # update map
            self.update_map(f"./maps/map{self.current_seed}", ".png")
            # store waypoints
            self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",
                                           delimiter=',')
        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]
        print(start_xy)
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)
    # --------------------------------------------------------------------------------------
    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)
    # --------------------------------------------------------------------------------------
    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
    # --------------------------------------------------------------------------------------                    
# =========================================================================================================================