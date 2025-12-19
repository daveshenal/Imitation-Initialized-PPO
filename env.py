'''
RL environment for training sctript
'''
import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import carla
from navigation import *
from penalty_calculations import *
from tensorflow.keras.models import load_model # type: ignore

HEIGHT  = 240
WIDTH   = 320

SPIN = 5 # angle of random spin
MAX_SPEED = 80    # 80kmh
SPEED_THRESHOLD = 2 # defines when we get close to desired speed so we drop the throttle
MAX_WP_DIST = 500 # 500m

HEIGHT_REQUIRED_PORTION = 0.5 # bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION  = 0.9
CAMERA_POS_Z = 1.3 
CAMERA_POS_X = 1.4

SHOW_PREVIEW = True  # simulator veiw
SHOW_CAM     = False # seg camera veiw
MAX_STEPS    = 1000
PREFERRED_SPEED = 40

WP_TYPE_MAPPING = {
		0: [1, 0],  # normal
		1: [0, 1],  # destination
}

STEERING_VALUES = (-0.9, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.9)


model_path = 'model_saved_from_CNN.h5' # CNN model path

class CarEnv(gym.Env):
	''' 
    Environment Initialization:
    
    CarEnv class inherits from gym.Env and initializes the environment with the CARLA client, vehicle model, and camera setup.
    camera provides semantic segmentation images to help the RL agent understand its environment visually,
    CNN model loaded from a pre-trained Keras model processes the camera input to generate the observation for the agent.
    '''

	front_camera = None
	
	def __init__(self):
		super(CarEnv, self).__init__()
  
		self.sensor_list = []  # Store sensor objects
  
        # Define action and observation space
        # They must be gym.spaces objects
		self.action_space = spaces.MultiDiscrete([9])

		self.height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
		self.width_from  = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
		self.width_to    = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

		self.observation_space = spaces.Dict({
			'img' : spaces.Box(low=0.0, high=1.0, shape=(7, 18, 8), dtype=np.float32), # Segmentation image
			'cspd': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),       # Current speed
			'dist': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),       # Normalized distance
			'ang' : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),      # Angle
			'typ': spaces.MultiBinary(2)                                               # Waypoint type
		})
		
		self.current_step = 0  # Current step count

		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(10.0)
		self.world  = self.client.get_world()
		# self.client.load_world('Town01')
		self.map = self.world.get_map()
		self.settings = self.world.get_settings()
		self.settings.no_rendering_mode = not SHOW_PREVIEW
		self.world.apply_settings(self.settings)
  
		self.spawn_points = self.map.get_spawn_points() 
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		self.cnn_model = load_model(model_path,compile=False)
		self.cnn_model.compile()
		if SHOW_PREVIEW:
			self.spectator = self.world.get_spectator()
   
       
	def reset(self):
		''' 
        reinitializes the environment.
        
        '''
		self.cleanup() # clean env
		self.current_step = 0  # reset the step counter

		self.collision_hist = []
		self.lane_invade_hist = []
  
		self.route, spawn_point = path_planner(self.spawn_points, self.map)  # complete route
		self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
		# route wont start with the passed start point (spawn point) by default GRP in Carla
		# so we have to move the vehicle to starting point of the route 
		trans = self.route[0][0].transform

		# apply a random yaw adjustment
		trans.rotation.yaw += random.randrange(-SPIN, SPIN, 1)
		# Set the modified transform to the vehicle
		self.vehicle.set_transform(trans)
		time.sleep(0.2)
  
  
		self.path_waypoints = get_path_waypoints(self.route) # path main wps
		currnet_wp_data = update_waypoint(self.path_waypoints, 0, trans.location)
		self.current_wp_idx = currnet_wp_data[1]
  
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{WIDTH}")
		self.sem_cam.set_attribute("image_size_y", f"{HEIGHT}")
		self.sem_cam.set_attribute("fov", f"90")
		
		camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(0.2)
  
		# showing camera at the spawn point
		if SHOW_CAM:
			cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Sem Camera', self.front_camera)
			cv2.waitKey(1)
						
		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
		self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.lanesensor)
		self.lanesensor.listen(lambda event: self.lane_data(event))

		while self.front_camera is None:
			time.sleep(0.01)

  
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
  
		self.cspd = 0.0
		# self.tl - not changed
  
		self.dist = currnet_wp_data[4]
		self.ang = get_angle(trans.rotation.yaw, currnet_wp_data[0].transform.rotation.yaw)
		self.type = currnet_wp_data[2]
  
		self.start_dist = self.dist # copy start dist for reward calculations
  
		return self.create_observation()

	def create_observation(self):
		# Normalize distance
		norm_dist = min(self.dist / MAX_WP_DIST, 1.0)

		# Normalize current speed
		norm_cspd = min(self.cspd / MAX_SPEED, 1.0)

		# Normalize angle using sine and cosine
		norm_angle_sin = np.sin(np.radians(self.ang))
		norm_angle_cos = np.cos(np.radians(self.ang))

		# One-hot encode waypoint type
		wp_type_one_hot = WP_TYPE_MAPPING[self.type] # intersection
  
		observation = {
			'img' : self.image_for_CNN,  # Segmentation image
			'cspd': np.array([norm_cspd], dtype=np.float32),       # Current speed
			'dist': np.array([norm_dist], dtype=np.float32),       # Distance
			'ang' : np.array([norm_angle_sin, norm_angle_cos], dtype=np.float32), # Angle
			'typ' : np.array(wp_type_one_hot, dtype=np.float32)    # Waypoint type\
		}
  
		return observation


	def step(self, action):
		''' 
		step function maps actions to steering angles and applies vehicle control for throttle and steering.
			
		rewards are calculated based on distance traveled, collisions, lane invasions, and steering lock, which 
		penalizes prolonged sharp turns.

		'''
		trans = self.vehicle.get_transform()
		if SHOW_PREVIEW:
			self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),carla.Rotation(yaw =-180, pitch=-90)))

		# Extracting actions
		str = STEERING_VALUES[action[0]]
  
		# Get the current speed of the vehicle
		velocity = self.vehicle.get_velocity()
		self.cspd = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
		estimated_throttle = self.maintain_speed(self.cspd)
		# map throttle and apply steer and throttle	
		self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=str, brake = 0.0))

		# storing camera to return at the end in case the clean-up function destroys it
		cam = self.front_camera
		# showing image
		if SHOW_CAM:
			cv2.imshow('Sem Camera', cam)
			cv2.waitKey(1)
  
		# start defining reward from each step
		reward = 0
		done = False # conditional fail
		truncated = False # max step reached
  
		# Increment the step counter
		self.current_step += 1 
  
		truncated = self.current_step >= MAX_STEPS
		if truncated:
			print("ms")
			done = True
  
		current_loc = self.vehicle.get_location()
		currnet_wp_data = update_waypoint(self.path_waypoints, self.current_wp_idx, current_loc)
		self.current_wp_idx = currnet_wp_data[1]
		if currnet_wp_data[3]:
			self.start_dist = currnet_wp_data[4]
		# Destination reached - end episode with large reward
		if currnet_wp_data[5]:  
			reward += 500
			print('D')
			done = True
   
     
		vehicle_yaw = trans.rotation.yaw
		dist_and_angle = get_dis_ang_from_nearest_wp(current_loc, vehicle_yaw, self.route)
  
		# rewards
		reward += calculate_distance_to_waypoint_reward(currnet_wp_data[3], currnet_wp_data[4], self.start_dist)
		penalty_1, done = calculate_penalty_angle(done, dist_and_angle[1])
		penalty_2, done = calculate_penalty_distance(done, dist_and_angle[0])
		penalty_3, done = collision_penalty(done, self.collision_hist)
		penalty_4, done = lane_invasion_penalty(done, self.lane_invade_hist)
  
		reward += (penalty_1 + penalty_2 + penalty_3 + penalty_4)

		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
		
		# self.tl - not changed
		self.dist = currnet_wp_data[4]
		self.ang = get_angle(vehicle_yaw, currnet_wp_data[0].transform.rotation.yaw)
		self.type = currnet_wp_data[2]
  
		observation = self.create_observation()

		return observation, reward, done, {'truncated': truncated}

	
	def maintain_speed(self,s):
		''' 
		this is a very simple function to maintan desired speed
		s arg is actual current speed
		'''
		if s >= PREFERRED_SPEED:
			return 0
		elif s < PREFERRED_SPEED - PREFERRED_SPEED:
			return 0.7 # think of it as % of "full gas"
		else:
			return 0.3 # tweak this if the car is way over or under preferred speed 

	def cleanup(self):
		''' 
        function to cealn carla env
        destroy all the actors in the env
        '''
		actors = self.world.get_actors()
		
		# Destroy all sensors stored in the list
		for sensor in self.sensor_list:
			sensor.destroy()
		# Clear the list after destruction
		self.sensor_list.clear()
				
		# Destroy all vehicles
		for vehicle in actors.filter('*vehicle*'):
				vehicle.destroy()

		# Final tick to process in CARLA
		self.world.tick()
		# # Close OpenCV windows
		# cv2.destroyAllWindows()

	def apply_cnn(self,im):
		''' 
        apply_cnn
        '''
		img = np.float32(im)
		img = img /255
		img = np.expand_dims(img, axis=0)
		cnn_applied = self.cnn_model([img,0],training=False)
		cnn_applied = np.squeeze(cnn_applied)
		return  cnn_applied ##[0][0]
 
	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		i = np.array(image.raw_data)
		i = i.reshape((HEIGHT, WIDTH, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
		self.front_camera = i

	def collision_data(self, event):
		self.collision_hist.append(event)
	def lane_data(self, event):
		self.lane_invade_hist.append(event)