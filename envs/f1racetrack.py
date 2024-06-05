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
import os
import numpy as np
import yaml
from argparse import Namespace
import gym
from f110gym.envs import linear_velocities_to_kph
from f110gym.envs.base_classes import Integrator
from mllib import CFileStore, CAutoObject
from f110gym.envs.f110_env import F110Env
from PIL import Image, ImageOps
from models.trajectory import nearest_waypoint_on_trajectory
from pyglet.gl import GL_POINTS

# =========================================================================================================================
class F1RaceTrack(object):
  MAP_EXTENSION = ".png"
  
  # --------------------------------------------------------------------------------------
  def __init__(self, track_folder, is_fast_rendering=True, is_recording=False, track_name=None, car_params=None, time_step=0.01):
    self.track_fs     = CFileStore(track_folder)
    
    if track_name is None:
      self.track_name = os.path.basename(os.path.normpath(track_folder))
    else:
      self.track_name = track_name
    self.env          = None
    self.config       = None
    self.init_vector  = None
    self.purse_pursuit_planner = None
    
    
    self.is_fast_rendering = is_fast_rendering
    self.is_recording = is_recording
    

    if self.track_fs.Exists("config_example_map.yaml"):
      with open(self.track_fs.File("config_example_map.yaml")) as file:
          conf_dict = yaml.load(file, Loader=yaml.FullLoader)
      self.config = Namespace(**conf_dict)
      
    else:
      # x_m, y_m, w_tr_right_m, w_tr_left_m
      
      # Raceline .csv file location and format metadata. Start position and orientation (None to autoconfig)
      dFieldValues = {
                         "wpt_path": "."
                        ,"wpt_delim": ";"
                        ,"wpt_rowskip": 3
                        ,"wpt_xind" : 1
                        ,"wpt_yind" : 2
                        ,"wpt_thind": 3
                        ,"wpt_vind" : 5
                        ,"sx": None
                        ,"sy": None
                        ,"stheta": None
                     }
      self.config = CAutoObject(dFieldValues)
      
    self.config.wpt_path = self.track_fs.File(self.track_name + "_raceline.csv")
    self.raceline = np.loadtxt(self.config.wpt_path, 
                              delimiter=self.config.wpt_delim, 
                              skiprows=self.config.wpt_rowskip)
    # Raceline file header
    #"x_m"         : The x-coordinate of a point on the raceline.
    #"y_m"         : The y-coordinate of a point on the raceline.
    #"theta"       : The orientation angle of the raceline at the given point.
    #"vx_mps"      : The desired velocity of the vehicle at the given point.
    #"kappa_radpm" : The curvature of the raceline at the given point, measured in radians per meter.
    
    #    ;xind;yind;   thind;        vint;
    #   0;   1;   2;       3;           4;      5;       6
    # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    self.waypoints = np.concatenate([  self.raceline[:, self.config.wpt_xind][...,np.newaxis], 
                                       self.raceline[:, self.config.wpt_yind][...,np.newaxis]
                                    ], axis=1)
    
    if self.config.sx is not None:
      init_pos = np.array([self.config.sx, self.config.sy])
    else:
      self.config.sx = 0
      self.config.sy = 0
      init_pos = np.zeros(2)
    nearest_point, nearest_dist, t, i = nearest_waypoint_on_trajectory(init_pos, self.waypoints)
    
    self.config.sx, self.config.sy = nearest_point
    
    
    dY = self.waypoints[i+1,1]-self.waypoints[i,1]
    dX = self.waypoints[i+1,0]-self.waypoints[i,0]
    self.config.stheta = np.arctan2(dY, dX)
    
    print(f"Initializing Vehicle")
    print(f"  |__  orientation:{(self.config.stheta*180)/np.pi:.1f}degs")
    print(f"  |___ starting wp:{i} total wps:{len(self.waypoints)}") 
    print(f"  |___ coords:{nearest_point}")
    
    self.car_params = car_params
    if self.car_params is not None:
      self.env = gym.make('f110gym:f110-v0', 
                          map=self.track_fs.File(self.track_name + "_map"),
                          map_ext=F1RaceTrack.MAP_EXTENSION, 
                          num_agents=1, 
                          params=self.car_params,
                          timestep=time_step,              # Not present at training.py
                          integrator=Integrator.RK4)  # Not present at training.py
    else:
      self.env = gym.make('f110gym:f110-v0', 
                          map=self.track_fs.File(self.track_name + "_map"),
                          map_ext=F1RaceTrack.MAP_EXTENSION, 
                          num_agents=1, 
                          timestep=time_step,              # Not present at training.py
                          integrator=Integrator.RK4)  # Not present at training.py                       
                                               
    self.env.is_visualizing = self.is_recording
    self.env.add_render_callback(self.render_callback)
        
    # starting pose for map
    self.init_vector = None
    
    if self.config.stheta is not None:
      self.init_vector =  np.array([[self.config.sx, self.config.sy, self.config.stheta]])
      #self.init_vector =  np.array([[0.7, 0.0, 1.37079632679]])

    # store car dimensions and some track info
    self.car_length = self.env.params['length']
    self.car_width = self.env.params['width']      
    self.track_width = 3.2  # ~= track width, see random_trackgen.py

    # radius of circle where car can start on track, relative to a centerpoint
    self.start_radius = (self.track_width / 2) - \
        ((self.car_length + self.car_width) / 2)  # just extra wiggle room
    
    self._initial_rotation = None
    self.drawn_waypoints = []
    self.last_state = None
  # --------------------------------------------------------------------------------------
  def vehicle_speed(self, state):
    linear_velocity_x = state["linear_vels_x"][0]
    linear_velocity_y = state["linear_vels_y"][0]
    vspeed_kmh = linear_velocities_to_kph(linear_velocity_x, linear_velocity_y)*self.env.speed_adjust_to_real
            
    return vspeed_kmh
  # --------------------------------------------------------------------------------------
  def get_nearest_waypoint(self, x, y):
    init_pos = np.array([x, y])
    nearest_point, nearest_dist, t, i = nearest_waypoint_on_trajectory(init_pos, self.waypoints)
    return i, nearest_point, nearest_dist
  # --------------------------------------------------------------------------------------
  @property
  def screenshot(self):
    oImage = Image.fromarray(F110Env.renderer.screenshot)    
    oImage = ImageOps.flip(oImage)
    #oImage = oImage.resize((nImage.shape[1]//2, nImage.shape[0]//2))
    return np.array(oImage)
  # --------------------------------------------------------------------------------------
  @property
  def initial_rotation(self):
    return self._initial_rotation
  # --------------------------------------------------------------------------------------
  @initial_rotation.setter
  def initial_rotation(self, degrees):
    self._initial_rotation = degrees
    #if self.init_vector is None:
    # start from origin if no pose input
    start_xy = np.zeros(2)
    if self.config.sx is not None:
      start_xy[0] = self.config.sx
    if self.config.sy is not None:
      start_xy[1] = self.config.sy
      
    if self._initial_rotation is None:
      direction = np.random.uniform(0, 2 * np.pi)
    else:
      direction = np.pi*(self._initial_rotation/180.0)
    '''
    # get slope perpendicular to track direction
    slope = np.tan(direction + np.pi / 2)
    # get magintude of slope to normalise parametric line
    magnitude = np.sqrt(1 + np.power(slope, 2))
    # get random point along line of width track
    rand_offset = np.random.uniform(-1, 1)
    rand_offset_scaled = rand_offset * self.start_radius

    # convert position along line to position between walls at current point
    x, y = start_xy + rand_offset_scaled * np.array([1, slope]) / magnitude

    # point car in random forward direction, not aiming at walls
    t = -np.random.uniform(max(-rand_offset * np.pi / 2, 0) - np.pi / 2,
                           min(-rand_offset * np.pi / 2, 0) + np.pi / 2) + direction
                          
    #np.array([[self.config.sx, self.config.sy, self.config.stheta]])               
    self.init_vector =  np.array([[x, y, t]])
    '''
    x, y = start_xy
    y = y - 0.9
    
    t = direction
    self.init_vector =  np.array([[x, y, t]])
  # --------------------------------------------------------------------------------------
  def reset(self, init_vector=None, pure_pursuit_planner=None):
    if init_vector is not None:
      self.init_vector = init_vector
                        
    self.pure_pursuit_planner = pure_pursuit_planner
    self.last_state = self.env.reset(self.init_vector)
    return self.last_state
  # --------------------------------------------------------------------------------------
  def step(self, action):
    self.last_state = self.env.step(action)
    return self.last_state
  # --------------------------------------------------------------------------------------
  def render(self, mode=None, **kwargs):
    if self.is_fast_rendering:
      return self.env.render(mode="human_fast", **kwargs)
    else:
      return self.env.render(mode="human", **kwargs)
  # --------------------------------------------------------------------------------------    
  def render_callback(self, env_renderer):
      # custom extra drawing function

      e = env_renderer

      # update camera to follow car
      x = e.cars[0].vertices[::2]
      y = e.cars[0].vertices[1::2]
      top, bottom, left, right = max(y), min(y), min(x), max(x)
      e.score_label.x = left
      e.score_label.y = top - 700
      e.left = left - 800
      e.right = right + 800
      e.top = top + 800
      e.bottom = bottom - 800
      
      #if self.pure_pursuit_planner is not None:
      #  self.pure_pursuit_planner.render_waypoints(env_renderer)
      #else:
      self.render_racing_line(env_renderer)
  # --------------------------------------------------------------------------------------
  def render_racing_line(self, env_renderer):
        """
        update waypoints being drawn by EnvRenderer
        """
        #points = self.waypoints

        points = self.waypoints#np.vstack((self.waypoints[:, 0], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = env_renderer.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]        
  # --------------------------------------------------------------------------------------
  @property
  def action_space(self):
    return self.env.action_space
  @property
  def observation_space(self):
    return self.env.observation_space
  @property
  def reward_range(self):
    return self.env.reward_range
  @property  
  def metadata(self):
    return self.env.metadata
  @property
  def spec(self):
    return self.env.spec
  @property
  def compute_reward(self):
    return self.env.compute_reward
  @property
  def unwrapped(self):
    return self.env.unwrapped
  @property
  def params(self):
    return self.env.params
          
# =========================================================================================================================  