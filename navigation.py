# route planning
import sys
import random

sys.path.append('C:/CARLA_0.9.14/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner # type: ignore

def path_planner(spawn_points, map):
    """
    Get the longest route waypoints from a starting point to various spawn points.

    Args:
        start_point (carla.Location): The starting location from which to trace routes.
        spawn_points (list[carla.SpawnPoint]): A list of spawn points to which routes will be traced.
        map (carla.Map): The Carla map object, used for route planning.

    Returns:
        list[tuple]: A list of waypoints representing the longest route found,
                     where each waypoint is a tuple containing a Carla waypoint object
                     and its associated location.
    """   
    sampling_resolution = 1
    grp = GlobalRoutePlanner(map, sampling_resolution)
    
    # intersections = (0, 99, 21, 59, 79, 51, 52, 102, 11, 12, 25, 24)
    # spawn_point = spawn_points[random.choice(intersections)]
    
    spawn_point = random.choice(spawn_points)
    
    # spawn_point = spawn_points[21]
    # end_point = spawn_points[45]
    

    # Now let's pick the longest possible route
    distance = 0
    for loc in spawn_points:
        cur_route = grp.trace_route(spawn_point.location, loc.location)
        if len(cur_route) > distance:
            distance = len(cur_route)
            route = cur_route
            
    # route = grp.trace_route(spawn_point.location, end_point.location)
            
    return route, spawn_point


def get_path_waypoints(route):
    """
    Extract the juctions start and end waypoints.

    Args:
        route (list[tuple]): A list of waypoints, where each waypoint is represented 
                             as a tuple containing a Carla waypoint object and its location.

    Returns:
        list[carla.Waypoint]: A list of waypoints
    """
    # List to hold waypoints of different types
    path_waypoints = []
    
    # Flag to track if the last waypoint is a junction
    last_wp_is_junction = route[-1][0].is_junction if route else False

    # Iterate over waypoints in the route
    for i in range(len(route)):
        if route[i][0].is_junction:
            # Check if it's the start of a junction waypoints set
            if (i == 0 or not route[i-1][0].is_junction):  # Start of a junction
                path_waypoints.append(route[i][0])
            # Check if it's the end of a 'j' set
            elif (i == len(route)-1 or not route[i+1][0].is_junction):  # End of a junction
                path_waypoints.append(route[i][0])
                
    # Add the last waypoint if it is not a junction
    if not last_wp_is_junction:
        path_waypoints.append(route[-1][0])

    
    return path_waypoints  # Return the list of waypoints


def update_waypoint(path_waypoints, current_wp_indx, current_loc):
    
    """
    Update to the next waypoint if the vehicle is close enough to the current one.
    
    returns,
        current_wp: next waypoint to reach. (not update if its the destination)
        wp_type: way point type (normal, destination, intersection, other)
        reached: boolean value to know that waypoint has reached
        dist = distance to new waypoint
    
    """
    threshold = 3 # 3m
    current_wp = path_waypoints[current_wp_indx]
    wp_type = 0 # normal by default
    reached = False # current waypoint reached
    destination = False # end of the path
            
    # Calculate distance to the current waypoint
    dist = current_loc.distance(current_wp.transform.location)
    
    # Check if the vehicle is within the threshold distance to update to the next waypoint
    if dist < threshold:
        reached = True
        if current_wp_indx < len(path_waypoints) - 1:
            current_wp_indx += 1  # Move to the next waypoint
            current_wp = path_waypoints[current_wp_indx]  # Update current waypoint
            dist = current_loc.distance(current_wp.transform.location)  # Calculate new distance
            if current_wp_indx == len(path_waypoints) - 1:
                wp_type = 1 # next waypoint is destination
        else:
            destination = True  # Current waypoint is the destination
    return current_wp, current_wp_indx, wp_type, reached, dist, destination


def get_dis_ang_from_nearest_wp(current_loc, vehicle_yaw, route):
    """
    Calculate the distance and angle difference between the vehicle and the nearest waypoint in the route.

    Args:
        vehicle (carla.Vehicle): The vehicle object.
        route (list[tuple]): A list of waypoints from the route, where each entry is a tuple
                             containing a CARLA Waypoint object and a RoadOption.

    Returns:
        tuple: A tuple containing the distance to the nearest waypoint and the absolute angular difference in yaw.
    """
    # Get the vehicle's current location
    selected_wp, dist_to_wp = get_closest_wp(current_loc, route)

    # Check if we found a valid waypoint
    if selected_wp:        
        dir_differ = get_angle(vehicle_yaw, selected_wp.transform.rotation.yaw)

        return dist_to_wp, dir_differ
    
    else:
        print("No waypoint found.")
        return 0.0, 0.0
    

def get_angle(vehicle_yaw, wp_yaw):    
    # Calculate the yaw difference
    dir_differ = vehicle_yaw - wp_yaw

    # Normalize to -180 to 180 degrees
    dir_differ = (dir_differ + 180) % 360 - 180
    return dir_differ

    
def get_closest_wp(current_loc, route):
    min_dist = float('inf')  # Initialize to a very large distance
    closest_wp = None  # To store the closest waypoint
    # Find the closest waypoint in the route
    for wp, _ in route:
        wp_loc = wp.transform.location
        dist = current_loc.distance(wp_loc)
        
        # Update if this waypoint is closer
        if dist < min_dist:
            min_dist = dist
            closest_wp = wp  # Store the closest waypoint object
            
    return closest_wp, min_dist 
            
            
            
                    
        
        


    




