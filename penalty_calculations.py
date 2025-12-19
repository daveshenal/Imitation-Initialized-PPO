# penalty_calculations.py

import carla

def calculate_distance_to_waypoint_reward(reached, current_dist, start_dist):
    """
    Calculate the reward based on the distance to the waypoint and stopping at the destination.
    
    args:
        done: waypoint reched
        current_dist: distance to waypoint
        start_dist: start_distance form the waypoint

   """ 
    # Reward scaling based on distance (closer is better)
    if reached:
        reward = 300.0  # Reward for getting very close to the waypoint
    else:
        reward = max(0.0, 10.0 - (current_dist / start_dist * 10.0))  # Scale reward
    
    return reward



def collision_penalty(done, collision_hist):
    penalty = 1
    if len(collision_hist) != 0:
        done = True
        penalty = -300
        print('c')
        return penalty, done
    return penalty, done


def lane_invasion_penalty(done, lane_invade_hist):
    """
    Calculate a penalty for lane invasions based on crossed lane markings.
    Stops checking as soon as an invalid marking is detected.
    """
    penalty = 1
    
    # Define restricted lane markings
    restricted_markings = {
        carla.LaneMarkingType.Solid,
        carla.LaneMarkingType.SolidSolid,
        carla.LaneMarkingType.SolidBroken
    }
    
    # Iterate through each LaneInvasionEvent in the history
    for event in lane_invade_hist:
        # Iterate through the list of crossed lane markings
        for marking in event.crossed_lane_markings:
            # Check if the marking type is in the restricted markings set
            if marking.type in restricted_markings:
                done = True
                penalty = -300
                print('l')
                return penalty, done  # Penalize and stop checking further

    return penalty, done  # No penalty if only permissible markings were crossed


# Penalties for not following a valid path

def calculate_penalty_angle(done, angle_difference):
    abgle_threshold = 1  # angle for no penalty
    angle_difference = abs(angle_difference) # ignor direction
    angle_penalty_weight = 30.0  # Weight for angle penalty
    penalty = 0
    
    if angle_difference > 18:
        penalty = -300
        print(f'a : {angle_difference}')
        done = True
    elif angle_difference > abgle_threshold:
        penalty = (angle_difference / 180) * angle_penalty_weight  # Normalize angle to a 0-1 scale
        penalty = -(penalty*penalty)
        
    return penalty,done
	
def calculate_penalty_distance(done, distance):
    # Define the threshold for minimal penalty
    distance_threshold = 0.5  # Distance in meters for no penalty
    distance_penalty_weight = 1.0  # Weight for distance penalty
    distance_penalty = 0
      
    if distance > 4:
        distance_penalty = -300
        done = True
        print(f'd : {distance}')
    # Calculate distance penalty only if outside the threshold
    elif distance > distance_threshold:
        distance_penalty = (distance - distance_threshold) * distance_penalty_weight
        distance_penalty = -(distance_penalty*distance_penalty)
        
    return distance_penalty,done

def calculate_speed_reward(done, PREFERRED_SPEED, s):
    dif = abs(PREFERRED_SPEED - s)
    prnalty = 0
    if dif < 5:
        prnalty = 5 - dif
    elif s > PREFERRED_SPEED:
        done = True
        prnalty = -300
    else:           
        prnalty = -dif
    
    return prnalty, done
    

        