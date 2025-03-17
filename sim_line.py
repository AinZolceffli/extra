import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import differential_evolution
import time


class TrafficSystem:
	def __init__(self):
		self.vehicles = []
		self.lights = []
		self.shape = None
		self.edges = []
		self.time = 0
		# For timed spawn tracking
		self.time_since_last_spawn = 0
		self.next_spawn_time = 0
	
	def line(self, length):
		"""Create a simple line with one edge of given length"""
		self.shape = "line"
		self.edges = [(0, 0, length, 0)]  # (x1, y1, x2, y2) format
		self.total_perimeter = length
	
	def line_with_light(self, length, light_positions):
		"""Create a line with traffic lights at specified positions"""
		self.shape = "line_with_light"
		self.edges = [(0, 0, length, 0)]
		self.total_perimeter = length
		# Add traffic lights at specified positions
		for i, pos in enumerate(light_positions):
			light = TrafficLight(i, "red", pos)
			self.add_traffic_light(light)
	
	def square(self, side_length):
		"""Create a closed square with 4 traffic lights at corners"""
		self.shape = "square"
		# Define the four edges of the square (x1, y1, x2, y2)
		self.edges = [
			(0, 0, side_length, 0),  # Bottom edge
			(side_length, 0, side_length, side_length),  # Right edge
			(side_length, side_length, 0, side_length),  # Top edge
			(0, side_length, 0, 0)  # Left edge
		]
		self.total_perimeter = 4 * side_length
		
		# Add traffic lights at the corners
		for i, corner in enumerate([(0, 0), (side_length, 0),
		                            (side_length, side_length), (0, side_length)]):
			light = TrafficLight(i, "red" if i % 2 == 0 else "green", corner)
			self.add_traffic_light(light)
	
	def grid_network(self, rows, cols, road_length):
		"""Create a grid network with intersections using a mesh representation"""
		self.shape = "grid"
		self.edges = []
		self.nodes = []  # Renamed from intersections to emphasize mesh/graph structure
		self.edge_to_endpoints = {}  # Maps edge index to its endpoints (for navigation)
		edge_id = 0
		
		# Create all intersection nodes first
		self.node_map = {}  # Maps (x,y) to node ID
		node_id = 0
		for i in range(rows + 1):
			for j in range(cols + 1):
				x = j * road_length
				y = i * road_length
				self.nodes.append((x, y))
				self.node_map[(x, y)] = node_id
				node_id += 1
		
		# Create horizontal edges (connections between nodes)
		for i in range(rows + 1):
			for j in range(cols):
				x1 = j * road_length
				y1 = i * road_length
				x2 = (j + 1) * road_length
				y2 = i * road_length
				self.edges.append((x1, y1, x2, y2))
				self.edge_to_endpoints[edge_id] = ((x1, y1), (x2, y2))
				edge_id += 1
		
		# Create vertical edges (connections between nodes)
		for i in range(rows):
			for j in range(cols + 1):
				x1 = j * road_length
				y1 = i * road_length
				x2 = j * road_length
				y2 = (i + 1) * road_length
				self.edges.append((x1, y1, x2, y2))
				self.edge_to_endpoints[edge_id] = ((x1, y1), (x2, y2))
				edge_id += 1
		
		# Create adjacency list for navigation (mesh connectivity)
		self.adjacency_list = {}
		for i, edge in enumerate(self.edges):
			x1, y1, x2, y2 = edge
			start_point = (x1, y1)
			end_point = (x2, y2)
			
			if start_point not in self.adjacency_list:
				self.adjacency_list[start_point] = []
			if end_point not in self.adjacency_list:
				self.adjacency_list[end_point] = []
			
			self.adjacency_list[start_point].append((end_point, i))
			# Bidirectional roads
			self.adjacency_list[end_point].append((start_point, i))
		
		# Add traffic lights at each node (intersection)
		for idx, (x, y) in enumerate(self.nodes):
			light = TrafficLight(idx, "red", (x, y))
			self.add_traffic_light(light)
		
		# Calculate total perimeter
		self.total_perimeter = road_length * (rows * (cols + 1) + cols * (rows + 1))
	
	def add_vehicle(self, vehicle):
		"""Add a vehicle to the traffic system"""
		self.vehicles.append(vehicle)
	
	def add_traffic_light(self, light):
		"""Add a traffic light to the traffic system"""
		self.lights.append(light)
	
	def update(self, dt=1, maintain_count=None):
		"""Update the system by dt time units"""
		self.time += dt
		self.time_since_last_spawn += dt
		
		# Update traffic lights
		for light in self.lights:
			light.update(dt)
		
		# Update vehicles
		vehicles_to_remove = []
		for vehicle in self.vehicles:
			# Vehicle update returns True if vehicle should be removed
			if vehicle.update(dt, self):
				vehicles_to_remove.append(vehicle)
		
		# Remove vehicles that have reached their destinations
		vehicles_removed = 0
		for vehicle in vehicles_to_remove:
			if vehicle in self.vehicles:
				self.vehicles.remove(vehicle)
				print(f"Vehicle {vehicle.id} removed after reaching destination")
				vehicles_removed += 1
		
		# If maintaining vehicle count, spawn new vehicles as needed
		if maintain_count is not None and vehicles_removed > 0:
			self.maintain_vehicle_count(maintain_count)
	
	def add_random_vehicles(self, count, velocity=1, length=1, minimum_distance=1.0):
		"""
		Add multiple vehicles with random start and destination nodes in a grid network.

		Parameters:
		-----------
		count : int
			Number of vehicles to add
		velocity : float
			Speed of the vehicles (default: 1)
		length : float
			Length of the vehicles (default: 1)
		minimum_distance : float
			Minimum safe distance between vehicles (default: 1.0)

		Returns:
		--------
		int
			Number of vehicles successfully added
		"""
		if self.shape != "grid" or not hasattr(self, 'nodes'):
			print("Random vehicle addition only works in grid networks")
			return 0
		
		# Get the next available ID
		next_id = max([v.id for v in self.vehicles], default=-1) + 1
		
		# Track how many vehicles were successfully added
		added_count = 0
		
		# Add vehicles
		for i in range(count):
			vehicle = Vehicle(next_id + i,
			                  velocity=velocity,
			                  length=length,
			                  minimum_distance=minimum_distance)
			
			# Try to set random start and destination
			success = vehicle.set_random_start_and_destination(self)
			if success:
				self.add_vehicle(vehicle)
				added_count += 1
			else:
				print(f"Failed to add vehicle {next_id + i}")
		
		print(f"Successfully added {added_count} vehicles")
		return added_count
	
	def maintain_vehicle_count(self, target_count, velocity=0.3, minimum_distance=1.0):
		"""
		Maintain a fixed count of vehicles by spawning new ones when others reach destinations.

		Parameters:
		-----------
		target_count : int
			The desired number of vehicles to maintain
		velocity : float
			Speed of new vehicles (default: 0.3)
		minimum_distance : float
			Minimum safe distance between vehicles (default: 1.0)
		"""
		# Check how many vehicles we need to add
		vehicles_to_add = target_count - len(self.vehicles)
		
		if vehicles_to_add <= 0:
			return 0  # Already at or above target count
		
		# Add new vehicles
		added_count = 0
		for i in range(vehicles_to_add):
			# Get next available ID
			next_id = max([v.id for v in self.vehicles], default=-1) + 1
			
			# Create new vehicle
			vehicle = Vehicle(next_id,
			                  velocity=velocity,
			                  length=1,
			                  minimum_distance=minimum_distance)
			
			# Try to set random start and destination
			success = vehicle.set_random_start_and_destination(self)
			if success:
				self.add_vehicle(vehicle)
				added_count += 1
			else:
				print(f"Failed to add replacement vehicle {next_id}")
		
		if added_count > 0:
			print(f"Added {added_count} new vehicles to maintain count of {target_count}")
		
		return added_count
	
	def spawn_vehicles_on_line(self, count, spawn_rate=1.0, velocity=1, length=1, minimum_distance=1.0):
		"""
		Spawn vehicles at the start of a line-based traffic system with a uniform probability.

		Parameters:
		-----------
		count : int
			Number of vehicles to spawn
		spawn_rate : float
			Probability of spawning a vehicle in each attempt (default: 1.0)
		velocity : float
			Speed of the vehicles (default: 1)
		length : float
			Length of the vehicles (default: 1)
		minimum_distance : float
			Minimum safe distance between vehicles (default: 1.0)

		Returns:
		--------
		int
			Number of vehicles successfully spawned
		"""
		if self.shape not in ["line", "line_with_light"]:
			print("This spawning method only works for line-based traffic systems")
			return 0
		
		# Get the next available vehicle ID
		next_id = max([v.id for v in self.vehicles], default=-1) + 1
		
		# Track spawned vehicles
		spawned_count = 0
		attempted_spawns = 0
		
		# Continue until we've spawned the requested number of vehicles
		# or we've attempted too many times
		while spawned_count < count and attempted_spawns < count * 10:
			attempted_spawns += 1
			
			# Check if spawn area is clear (no vehicle in the first 'length + minimum_distance' units)
			spawn_area_clear = True
			for vehicle in self.vehicles:
				# For lines, position is simply the distance along the line
				if vehicle.position < length + minimum_distance:
					spawn_area_clear = False
					break
			
			# If spawn area is clear and we pass the spawn rate check, add a new vehicle
			if spawn_area_clear and random.random() <= spawn_rate:
				# Create a new vehicle at the start of the line
				vehicle = Vehicle(next_id + spawned_count,
				                  velocity=velocity,
				                  length=length,
				                  minimum_distance=minimum_distance)
				
				# For a line, the route is simply from start (0) to end (edge length)
				edge_length = self.edges[0][2]  # x2 coordinate of the first edge
				vehicle.position = 0
				vehicle.destination = edge_length
				vehicle.edge_id = 0  # The line has only one edge with ID 0
				
				self.add_vehicle(vehicle)
				spawned_count += 1
				
				print(f"Spawned vehicle {vehicle.id} at position 0")
			
			# Small time increment to simulate the system's update cycle
			self.update(0.1)
		
		print(f"Successfully spawned {spawned_count} vehicles out of {count} requested")
		return spawned_count
	
	def spawn_vehicles_with_normal_distribution(self, simulation_time, mean_time_between_spawns,
	                                            variance, velocity=1, length=1, minimum_distance=1.0):
		"""
		Spawn vehicles at the start of a line-based traffic system using a normal distribution
		for the time between spawns.

		Parameters:
		-----------
		simulation_time : float
			Total time to run the simulation
		mean_time_between_spawns : float
			Mean time between vehicle spawns (in time units)
		variance : float
			Variance of the time between spawns
		velocity : float
			Speed of the vehicles (default: 1)
		length : float
			Length of the vehicles (default: 1)
		minimum_distance : float
			Minimum safe distance between vehicles (default: 1.0)

		Returns:
		--------
		int
			Number of vehicles successfully spawned
		"""
		if self.shape not in ["line", "line_with_light"]:
			print("This spawning method only works for line-based traffic systems")
			return 0
		
		# Reset the simulation time
		self.time = 0
		self.time_since_last_spawn = 0
		
		# Generate the first spawn time from the normal distribution
		# Use max to ensure we don't get negative times
		std_dev = np.sqrt(variance)
		self.next_spawn_time = max(0.1, np.random.normal(mean_time_between_spawns, std_dev))
		
		# Get the next available vehicle ID
		next_id = max([v.id for v in self.vehicles], default=-1) + 1
		
		# Track spawned vehicles
		spawned_count = 0
		
		# Run the simulation for the specified time
		while self.time < simulation_time:
			# Check if it's time to spawn a new vehicle
			if self.time_since_last_spawn >= self.next_spawn_time:
				# Check if spawn area is clear
				spawn_area_clear = True
				for vehicle in self.vehicles:
					if vehicle.position < length + minimum_distance:
						spawn_area_clear = False
						break
				
				if spawn_area_clear:
					# Create a new vehicle at the start of the line
					vehicle = Vehicle(next_id + spawned_count,
					                  velocity=velocity,
					                  length=length,
					                  minimum_distance=minimum_distance)
					
					# For a line, the route is simply from start (0) to end (edge length)
					edge_length = self.edges[0][2]  # x2 coordinate of the first edge
					vehicle.position = 0
					vehicle.destination = edge_length
					vehicle.edge_id = 0  # The line has only one edge with ID 0
					
					self.add_vehicle(vehicle)
					spawned_count += 1
					
					# Reset the time since last spawn
					self.time_since_last_spawn = 0
					
					# Generate the next spawn time from the normal distribution
					# Use max to ensure we don't get negative times
					self.next_spawn_time = max(0.1, np.random.normal((2 * mean_time_between_spawns), std_dev))
					
					print(f"Spawned vehicle {vehicle.id} at time {self.time:.2f}")
					print(f"Next spawn in {self.next_spawn_time:.2f} time units")
				else:
					# If we couldn't spawn, add a small delay before trying again
					self.time_since_last_spawn = self.next_spawn_time - 0.5
			
			# Update the system with a small time increment
			self.update(0.1)
		
		print(f"Simulation completed after {self.time:.2f} time units")
		print(f"Successfully spawned {spawned_count} vehicles")
		return spawned_count
	
	def spawn_vehicles_continuously_with_normal_distribution(self, mean_time_between_spawns,
	                                                         variance, velocity=1, length=1,
	                                                         minimum_distance=1.0):
		"""
		Initialize the system to spawn vehicles continuously during updates using a normal distribution
		for the time between spawns.

		Parameters:
		-----------
		mean_time_between_spawns : float
			Mean time between vehicle spawns (in time units)
		variance : float
			Variance of the time between spawns
		velocity : float
			Speed of the vehicles (default: 1)
		length : float
			Length of the vehicles (default: 1)
		minimum_distance : float
			Minimum safe distance between vehicles (default: 1.0)
		"""
		if self.shape not in ["line", "line_with_light"]:
			print("This spawning method only works for line-based traffic systems")
			return
		
		# Store the spawn parameters as system properties
		self.continuous_spawn = True
		self.spawn_mean = mean_time_between_spawns
		self.spawn_variance = variance
		self.spawn_velocity = velocity
		self.spawn_length = length
		self.spawn_min_distance = minimum_distance
		
		# Reset spawn tracking variables
		self.time_since_last_spawn = 0
		
		# Generate the first spawn time
		std_dev = np.sqrt(variance)
		self.next_spawn_time = max(0.1, np.random.normal(mean_time_between_spawns, std_dev))
		
		print(f"Continuous spawn mode activated")
		print(f"Mean time between spawns: {mean_time_between_spawns}")
		print(f"Standard deviation: {std_dev}")
		print(f"First vehicle will spawn in {self.next_spawn_time:.2f} time units")
	
	def optimize_traffic_lights(self, spawn_times, total_vehicles, simulation_function=None,
	                            velocity=14, min_green=20, max_green=40,
	                            min_red=6.67, max_red=13.33, max_iter=20, pop_size=10):
		"""
		Optimize traffic light timing using the integrated TrafficLightOptimizer.

		Parameters:
		-----------
		spawn_times : list
			Times when vehicles were spawned into the system
		total_vehicles : int
			Total number of vehicles that entered the system
		simulation_function : function, optional
			Custom function to run a simulation with given parameters.
			If None, will use run_simulation_for_optimization
		velocity : float
			Vehicle velocity in distance units per second
		min_green, max_green : float
			Minimum and maximum green light duration (seconds)
		min_red, max_red : float
			Minimum and maximum red light duration (seconds)
		max_iter : int
			Maximum number of iterations for optimization
		pop_size : int
			Population size for differential evolution

		Returns:
		--------
		dict: Optimization results
		"""
		# Extract traffic light positions
		light_positions = [light.position for light in self.lights]
		
		# Create an optimizer
		optimizer = TrafficLightOptimizer(
			light_positions=light_positions,
			velocity=velocity,
			min_green=min_green,
			max_green=max_green,
			min_red=min_red,
			max_red=max_red
		)
		
		# Use default simulation function if none provided
		if simulation_function is None:
			simulation_function = lambda *args, **kwargs: run_simulation_for_optimization(*args, **kwargs, system=self)
		
		# Run the optimization
		return optimizer.optimize(
			spawn_times=spawn_times,
			total_vehicles=total_vehicles,
			simulation_function=simulation_function,
			max_iter=max_iter,
			pop_size=pop_size
		)
	
	def detect_stops_at_traffic_lights(self):
		"""
		Function to detect and count how many vehicles stop at each traffic light.

		Returns:
		--------
		list: Number of stops at each traffic light
		"""
		# Initialize counters for stops at each light
		stops_per_light = [0] * len(self.lights)
		
		# Check each vehicle
		for vehicle in self.vehicles:
			if vehicle.waiting or vehicle.stopped:
				# Find the nearest traffic light
				nearest_light_idx = -1
				min_distance = float('inf')
				
				for i, light in enumerate(self.lights):
					# For line simulation, position is simple
					if hasattr(light, 'position') and hasattr(vehicle, 'position'):
						# Only count if vehicle is near but before the light
						distance = light.position - vehicle.position
						if 0 <= distance < min_distance:
							min_distance = distance
							nearest_light_idx = i
					# For 2D simulations
					elif hasattr(light, 'x') and hasattr(light, 'y'):
						# Get vehicle coordinates
						if hasattr(vehicle, 'get_coordinates'):
							vx, vy = vehicle.get_coordinates(self)
						else:
							# Fallback to _get_current_position if it exists
							if hasattr(vehicle, '_get_current_position'):
								vx, vy = vehicle._get_current_position(self)
							else:
								# Skip this vehicle if we can't get coordinates
								continue
						
						# Calculate distance
						distance = ((light.x - vx) ** 2 + (light.y - vy) ** 2) ** 0.5
						if distance < min_distance:
							min_distance = distance
							nearest_light_idx = i
				
				# If a nearby light was found, increment its stop counter
				if nearest_light_idx >= 0 and min_distance < 5.0:  # Count stops within 5 units of a light
					stops_per_light[nearest_light_idx] += 1
		
		return stops_per_light


class Vehicle:
	def __init__(self, id, position=0, length=1, velocity=1, minimum_distance=1.0, route=None):
		self.id = id
		self.position = position  # Position along the path (measured from start)
		self.length = length  # Length of the vehicle
		self.velocity = velocity  # Units per time step
		self.waiting = False  # If vehicle is waiting at a traffic light or behind another vehicle
		self.stopped = False  # If vehicle is stopped due to traffic
		self.minimum_distance = minimum_distance  # Minimum safe distance to the vehicle in front
		
		# For grid navigation
		self.current_edge = 0  # Current edge in route
		self.edge_progress = 0  # Progress along current edge (0-1)
		self.current_node = None  # Current intersection node
		self.next_node = None  # Next intersection to reach
		self.destination = None  # Final destination node
		self.reached_destination = False  # Flag to indicate if destination has been reached
		self.recently_visited = []  # Store last few visited nodes
	
	def set_random_start_and_destination(self, system):
		"""Randomly assign a start node and destination node for the vehicle"""
		if system.shape != "grid" or not system.nodes:
			return False
		
		# Choose random start and destination intersections
		available_nodes = system.nodes
		if len(available_nodes) < 2:
			return False
		
		# Select random start and destination nodes
		self.current_node = random.choice(available_nodes)
		# Make sure destination is different from start
		remaining_nodes = [node for node in available_nodes if node != self.current_node]
		self.destination = random.choice(remaining_nodes)
		self.destination_node = self.destination  # Add this line to store destination for visualization
		
		# Find an edge connected to the start node to begin on
		if self.current_node in system.adjacency_list:
			# Get a random edge connected to the current node
			next_options = system.adjacency_list[self.current_node]
			if next_options:
				# Choose the option that gets us closest to our destination
				best_option = self._choose_best_edge(system, next_options, self.destination)
				self.next_node = best_option[0]  # The node we're heading toward
				self.current_edge = best_option[1]  # The edge ID
				self.edge_progress = 0.0
				print(f"Vehicle {self.id}: Start node: {self.current_node}, destination: {self.destination}")
				print(f"Vehicle {self.id}: Initial edge {self.current_edge} to {self.next_node}")
				return True
		return False
	
	def _euclidean_distance(self, point1, point2):
		"""Calculate Euclidean distance between two points"""
		return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
	
	def _choose_best_edge(self, system, options, destination):
		"""Improved edge selection with direction preference and memory"""
		best_options = []
		best_distance = float('inf')
		
		# Define a persistent direction preference
		if not hasattr(self, 'direction_preference'):
			self.direction_preference = None
			self.consistent_moves = 0
		
		# Track visited locations with a longer history
		if not hasattr(self, 'visited_history'):
			self.visited_history = {}
		
		# Add current node to history with count
		if self.current_node in self.visited_history:
			self.visited_history[self.current_node] += 1
		else:
			self.visited_history[self.current_node] = 1
		
		for next_node, edge_id in options:
			# Calculate raw distance
			dist = self._euclidean_distance(next_node, destination)
			
			# Apply penalties for frequently visited nodes (anti-oscillation)
			visit_count = self.visited_history.get(next_node, 0)
			if visit_count > 0:
				# Penalize revisiting nodes - each visit adds 20% distance penalty
				dist *= (1 + (visit_count * 0.2))
			
			# Apply direction preference if we have one
			if self.direction_preference:
				dx = next_node[0] - self.current_node[0]
				dy = next_node[1] - self.current_node[1]
				# If moving in different direction than preference, apply small penalty
				if (dx * self.direction_preference[0] + dy * self.direction_preference[1]) <= 0:
					dist *= 1.1
			
			# Update best options
			if dist < best_distance:
				best_distance = dist
				best_options = [(next_node, edge_id)]
			elif dist == best_distance:
				best_options.append((next_node, edge_id))
		
		# Choose randomly among best options to avoid deterministic cycles
		if best_options:
			choice = random.choice(best_options)
			
			# Update direction preference
			dx = choice[0][0] - self.current_node[0]
			dy = choice[0][1] - self.current_node[1]
			if dx != 0 or dy != 0:
				self.direction_preference = (dx, dy)
			
			return choice
		
		return options[0] if options else None
	
	def update_grid(self, dt, system):
		"""Update vehicle position on mesh using Cartesian coordinates"""
		# If vehicle has reached its destination, don't move
		if self.reached_destination:
			return False
		
		# If we don't have a destination yet, set one
		if self.destination is None:
			success = self.set_random_start_and_destination(system)
			if not success:
				return False  # Couldn't set up navigation
		
		# If waiting at a traffic light
		if self.waiting:
			# Check for traffic light at current position
			light_is_green = False
			for light in system.lights:
				if (abs(light.x - self.current_node[0]) < 0.1 and
						abs(light.y - self.current_node[1]) < 0.1):
					if light.state == "green":
						light_is_green = True
						break
			
			# Even if light is green, check if higher priority vehicles are at the same node
			if light_is_green:
				if self._has_higher_priority_vehicle_at_node(system):
					# Continue waiting even if light is green - higher priority vehicle has precedence
					return False
				else:
					self.waiting = False
					# After no longer waiting, choose next edge if we don't have one
					if self.current_node in system.adjacency_list:
						self._select_next_edge(system)
			
			if self.waiting:
				return False
		
		# Move along current edge
		if self.current_edge >= len(system.edges):
			print(f"ERROR: Vehicle {self.id} has invalid edge index {self.current_edge}")
			self.reached_destination = True
			return False
		
		# Check if there's a vehicle ahead of us that we need to queue behind
		if self._check_vehicle_ahead(system):
			self.stopped = True
			return False
		else:
			self.stopped = False
		
		# Check if the vehicle is at a traffic light
		for light in system.lights:
			if self._is_at_light(light, system):
				# Check if the light is red
				if light.state == "red":
					self.waiting = True
					return False  # Don't move if waiting at red light
		
		# Get current edge details
		edge = system.edges[self.current_edge]
		x1, y1, x2, y2 = edge
		
		# Figure out direction we're traveling on this edge
		going_forward = (abs(self.current_node[0] - x1) < 0.1 and
		                 abs(self.current_node[1] - y1) < 0.1)
		
		# Get correct start and end based on our direction
		start_x, start_y = (x1, y1) if going_forward else (x2, y2)
		end_x, end_y = (x2, y2) if going_forward else (x1, y1)
		
		# Calculate edge length in Cartesian space
		edge_length = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
		
		# Now calculate movement
		distance_to_move = self.velocity * dt
		distance_along_edge = distance_to_move / edge_length
		new_progress = self.edge_progress + distance_along_edge
		
		# If we reach the end of this edge (next node)
		if new_progress >= 1.0:
			# We've reached the node at the end of this edge
			old_node = self.current_node
			self.current_node = (end_x, end_y)
			print(f"Vehicle {self.id}: Arrived at node {self.current_node} from {old_node}")
			
			# Check if we've reached our destination
			if (abs(self.current_node[0] - self.destination[0]) < 0.1 and
					abs(self.current_node[1] - self.destination[1]) < 0.1):
				print(f"Vehicle {self.id}: Reached destination at coordinates {self.destination}")
				self.reached_destination = True
				return True  # Signal to remove this vehicle
			
			# Check for traffic light at this node
			for light in system.lights:
				if (abs(light.x - self.current_node[0]) < 0.1 and
						abs(light.y - self.current_node[1]) < 0.1):
					if light.state == "red":
						self.waiting = True
						return False
			
			# Choose next edge to travel on
			if self.current_node in system.adjacency_list:
				options = system.adjacency_list[self.current_node]
				
				# Find our previous node (where we came from)
				previous_node = (start_x, start_y)
				
				# Filter out options that would take us backwards or to recently visited nodes
				forward_options = [opt for opt in options
				                   if not (abs(opt[0][0] - previous_node[0]) < 0.1 and
				                           abs(opt[0][1] - previous_node[1]) < 0.1) and
				                   opt[0] not in self.recently_visited[-3:]]  # Avoid last 3 nodes
				
				if not forward_options:
					# If dead end, consider all options
					forward_options = options
				
				# Choose best option
				if forward_options:
					best_option = self._choose_best_edge(system, forward_options, self.destination)
					self.next_node = best_option[0]
					self.current_edge = best_option[1]
					
					# Get the new edge
					new_edge = system.edges[self.current_edge]
					x1, y1, x2, y2 = new_edge
					
					# Always start from the end that matches the current node
					if abs(self.current_node[0] - x1) < 0.1 and abs(self.current_node[1] - y1) < 0.1:
						# Current node is at the start point of the edge (x1,y1)
						self.edge_progress = 0.0
					else:
						# Current node is at the end point of the edge (x2,y2)
						# Swap edge endpoints so we're always moving "forward"
						x1, y1, x2, y2 = x2, y2, x1, y1
						# Reset progress to start of edge after swapping
						self.edge_progress = 0.0
					
					print(
						f"Vehicle {self.id}: Moving from node {self.current_node} to node {self.next_node} via edge {self.current_edge}")
				else:
					# No options available
					print(f"Vehicle {self.id}: No options available at node {self.current_node}")
					self.reached_destination = True
					return True  # Signal to remove this vehicle
			else:
				# Dead end
				self.reached_destination = True
				return True  # Signal to remove this vehicle
		else:
			# Continue along current edge
			self.edge_progress = new_progress
		
		return False  # Don't remove the vehicle
	
	def _edge_connects(self, system, edge_id, node1, node2):
		"""Check if the given edge connects the two nodes"""
		if edge_id >= len(system.edges):
			return False
		
		edge = system.edges[edge_id]
		x1, y1, x2, y2 = edge
		
		# Check if the edge connects the two nodes (within a small tolerance)
		connects_node1 = ((abs(x1 - node1[0]) < 0.1 and abs(y1 - node1[1]) < 0.1) or
		                  (abs(x2 - node1[0]) < 0.1 and abs(y2 - node1[1]) < 0.1))
		connects_node2 = ((abs(x1 - node2[0]) < 0.1 and abs(y1 - node2[1]) < 0.1) or
		                  (abs(x2 - node2[0]) < 0.1 and abs(y2 - node2[1]) < 0.1))
		
		return connects_node1 and connects_node2
	
	def _get_current_position(self, system):
		"""Get current (x,y) position in the grid"""
		# If vehicle is waiting at a traffic light, use the current_node position directly
		if self.waiting and hasattr(self, 'current_node'):
			return self.current_node
		
		if not hasattr(self, 'current_edge') or self.current_edge >= len(system.edges):
			return (0, 0)
		
		edge = system.edges[self.current_edge]
		x1, y1, x2, y2 = edge
		
		# Determine the correct direction of travel
		if hasattr(self, 'current_node'):
			going_forward = (abs(self.current_node[0] - x1) < 0.1 and
			                 abs(self.current_node[1] - y1) < 0.1)
			
			# Interpolate based on direction of travel
			if going_forward:
				x = x1 + self.edge_progress * (x2 - x1)
				y = y1 + self.edge_progress * (y2 - y1)
			else:
				x = x2 + self.edge_progress * (x1 - x2)
				y = y2 + self.edge_progress * (y1 - y2)
		else:
			# Default to forward direction if we don't have current_node
			x = x1 + self.edge_progress * (x2 - x1)
			y = y1 + self.edge_progress * (y2 - y1)
		
		return (x, y)
	
	def update(self, dt, system):
		"""
		Update the vehicle's position based on the traffic system
		Returns True if the vehicle should be removed from the system
		"""
		if system.shape == "grid":
			return self.update_grid(dt, system)
		
		if self.waiting:
			# Vehicle is waiting at a traffic light
			# Check if the light is now green
			for light in system.lights:
				if self._is_at_light(light, system) and light.state == "green":
					self.waiting = False
			
			# If still waiting, don't move
			if self.waiting:
				return False
		
		# Calculate the new position
		new_position = self.position + self.velocity * dt
		
		# Handle wrapping around in closed shapes
		if system.shape == "square":
			if new_position > system.total_perimeter:
				new_position = new_position % system.total_perimeter
		# Check if vehicle has reached the end of a line (non-closed shape)
		elif (system.shape == "line" or system.shape == "line_with_light") and new_position >= system.total_perimeter:
			print(f"Vehicle {self.id} has reached the end of the road at position {new_position}")
			return True  # Signal that vehicle should be removed
		
		# Check for traffic lights before moving
		for light in system.lights:
			light_position = self._get_light_position_on_path(light, system)
			if self._would_cross_light(self.position, new_position, light_position, system) and light.state == "red":
				# Stop at the light
				new_position = light_position
				self.waiting = True
				break
		
		# Check for collision with other vehicles
		for other in system.vehicles:
			if other is not self:
				# Determine if we are too close to the other vehicle
				if self._is_too_close(new_position, other, system):
					# Stop to maintain a safe distance
					new_position = self.position
					self.stopped = True
					break
				else:
					self.stopped = False
		
		self.position = new_position
		return False  # Don't remove the vehicle
	
	def _is_at_light(self, light, system):
		"""Check if vehicle is at a traffic light"""
		light_position = self._get_light_position_on_path(light, system)
		return abs(self.position - light_position) < 0.1
	
	def _would_cross_light(self, current_pos, new_pos, light_position, system):
		"""Check if moving from current_pos to new_pos crosses the light"""
		# Handle wrap-around for closed shapes
		if system.shape == "square":
			if new_pos < current_pos:
				return current_pos <= light_position or light_position <= new_pos
			return current_pos <= light_position <= new_pos
		return current_pos <= light_position <= new_pos
	
	def _get_light_position_on_path(self, light, system):
		"""Convert traffic light coordinates to a position along the path"""
		if system.shape == "line" or system.shape == "line_with_light":
			return light.position
		
		# For square, find distance along perimeter
		distance = 0
		for edge in system.edges:
			x1, y1, x2, y2 = edge
			if (light.x == x1 and light.y == y1):
				return distance
			edge_length = self._edge_length(edge)
			distance += edge_length
		return 0
	
	def _get_current_edge_position(self, system):
		"""Get the current edge and position on that edge"""
		if system.shape == "line" or system.shape == "line_with_light":
			return 0, self.position
		
		# For square, find which edge the vehicle is on
		distance = self.position
		for i, edge in enumerate(system.edges):
			edge_length = self._edge_length(edge)
			if distance <= edge_length:
				return i, distance
			distance -= edge_length
		return 0, 0
	
	def _get_edge_position(self, position, system):
		"""Get edge index and position on that edge for a given overall position"""
		if system.shape == "line" or system.shape == "line_with_light":
			return 0, position
		
		# For square, find which edge the position is on
		distance = position % system.total_perimeter
		for i, edge in enumerate(system.edges):
			edge_length = self._edge_length(edge)
			if distance <= edge_length:
				return i, distance
			distance -= edge_length
		return 0, 0
	
	def _edge_length(self, edge):
		"""Calculate length of an edge"""
		x1, y1, x2, y2 = edge
		return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # CORRECT formula
	
	def get_coordinates(self, system):
		"""Get the actual (x,y) coordinates of the vehicle"""
		if system.shape == "line" or system.shape == "line_with_light":
			return self.position, 0
		
		# For square, find coordinates based on position along perimeter
		edge_idx, distance = self._get_current_edge_position(system)
		x1, y1, x2, y2 = system.edges[edge_idx]
		
		# Calculate position along this edge
		total_edge_length = self._edge_length(system.edges[edge_idx])
		ratio = distance / total_edge_length if total_edge_length > 0 else 0
		x = x1 + ratio * (x2 - x1)
		y = y1 + ratio * (y2 - y1)
		return x, y
	
	def _is_too_close(self, new_pos, other_vehicle, system):
		"""Check if moving to new_pos would be too close to other_vehicle"""
		# For square shapes, vehicles have coordinates instead of just position
		if system.shape == "square":
			# Get coordinates for both vehicles
			self_coords = self._get_coordinates_at_position(new_pos, system)
			other_coords = other_vehicle.get_coordinates(system)
			
			# Calculate distance between vehicles (simple Euclidean distance)
			self_x, self_y = self_coords
			other_x, other_y = other_coords
			distance = ((self_x - other_x) ** 2 + (self_y - other_y) ** 2) ** 0.5
			
			# If distance is less than minimum distance, they are too close
			return distance < self.minimum_distance
		else:
			# For line shapes, only consider vehicles that are ahead of the current vehicle
			if other_vehicle.position <= self.position:
				return False
			
			# Simple check if the new position overlaps with other vehicle's position
			# The front of this vehicle (new_pos) should not be between other vehicle's rear and front
			other_rear = other_vehicle.position - other_vehicle.length
			# Check if new position would be between the other vehicle's rear and its position
			return other_rear - new_pos <= self.minimum_distance
	
	def _get_coordinates_at_position(self, position, system):
		"""Calculate coordinates at a given position on the road network"""
		# For square shape
		if system.shape == "square":
			# Find the edge the vehicle is on
			current_pos = position % system.total_perimeter
			edge_start = 0
			
			for edge in system.edges:
				x1, y1, x2, y2 = edge
				edge_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # ENSURE this is correct
				
				if edge_start <= current_pos < edge_start + edge_length:
					# Vehicle is on this edge
					# Calculate how far along the edge (as a fraction)
					fraction = (current_pos - edge_start) / edge_length
					# Interpolate between start and end points
					x = x1 + fraction * (x2 - x1)
					y = y1 + fraction * (y2 - y1)
					return (x, y)
				
				edge_start += edge_length
			
			# If we get here, there's an error
			return (0, 0)
		else:
			# For line shape, x is the position, y is 0
			return (position, 0)
	
	def _select_next_edge(self, system):
		"""Helper to select the next edge to travel on"""
		options = system.adjacency_list[self.current_node]
		
		# Find our previous node (where we came from) if available
		previous_node = None
		if hasattr(self, 'current_edge') and self.current_edge < len(system.edges):
			edge = system.edges[self.current_edge]
			x1, y1, x2, y2 = edge
			previous_node = (x1, y1) if (abs(self.current_node[0] - x2) < 0.1 and
			                             abs(self.current_node[1] - y2) < 0.1) else (x2, y2)
		
		# Filter out options that would take us backwards or to recently visited nodes
		if previous_node:
			forward_options = [opt for opt in options
			                   if not (abs(opt[0][0] - previous_node[0]) < 0.1 and
			                           abs(opt[0][1] - previous_node[1]) < 0.1) and
			                   opt[0] not in self.recently_visited[-3:]]
		else:
			forward_options = options
		
		if not forward_options:
			forward_options = options
		
		# Choose best option
		if forward_options:
			best_option = self._choose_best_edge(system, forward_options, self.destination)
			self.next_node = best_option[0]
			self.current_edge = best_option[1]
			
			# Set up the edge for travel
			self._setup_edge_travel(system)
	
	def _setup_edge_travel(self, system):
		"""Setup the vehicle to travel along the current edge"""
		# Get the new edge
		new_edge = system.edges[self.current_edge]
		x1, y1, x2, y2 = new_edge
		
		# Always start from the end that matches the current node
		if abs(self.current_node[0] - x1) < 0.1 and abs(self.current_node[1] - y1) < 0.1:
			# Current node is at the start point of the edge (x1,y1)
			self.edge_progress = 0.0
		else:
			# Current node is at the end point of the edge (x2,y2)
			# Swap edge endpoints so we're always moving "forward"
			x1, y1, x2, y2 = x2, y2, x1, y1
			# Reset progress to start of edge after swapping
			self.edge_progress = 0.0
		
		# Store edge coordinates for movement calculations
		self.edge_start = (x1, y1)
		self.edge_end = (x2, y2)
		
		# Log the movement
		print(
			f"Vehicle {self.id}: Moving from node {self.current_node} to node {self.next_node} via edge {self.current_edge}")
	
	def _check_vehicle_ahead(self, system):
		"""Check if there are vehicles ahead on the same edge or at the next node"""
		if not hasattr(self, 'current_edge') or self.current_edge >= len(system.edges):
			return False
		
		# Get current edge and figure out our direction on this edge
		edge = system.edges[self.current_edge]
		x1, y1, x2, y2 = edge
		
		# Determine our direction of travel
		going_forward = (abs(self.current_node[0] - x1) < 0.1 and
		                 abs(self.current_node[1] - y1) < 0.1)
		
		# Get our target node (where we're heading)
		target_node = (x2, y2) if going_forward else (x1, y1)
		
		# Check for vehicles at the target intersection
		for other_vehicle in system.vehicles:
			if other_vehicle is self:
				continue
			
			# Check if the other vehicle is waiting at our target intersection
			if (hasattr(other_vehicle, 'waiting') and other_vehicle.waiting and
					hasattr(other_vehicle, 'current_node') and other_vehicle.current_node == target_node):
				return True
			
			# Check for vehicles that just left our current node - enforce priority
			if (hasattr(other_vehicle, 'current_edge') and
					other_vehicle.current_edge == self.current_edge and
					other_vehicle.id < self.id):  # Lower ID = higher priority
				
				# If vehicles are traveling in same direction, enforce minimum distance
				other_going_forward = (abs(other_vehicle.current_node[0] - x1) < 0.1 and
				                       abs(other_vehicle.current_node[1] - y1) < 0.1)
				
				if going_forward == other_going_forward:
					edge_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
					
					# If same direction, check progress
					if going_forward and other_vehicle.edge_progress > self.edge_progress:
						# Other vehicle is ahead on same path
						distance = (other_vehicle.edge_progress - self.edge_progress) * edge_length
						if distance < self.minimum_distance:
							return True
					elif not going_forward and other_vehicle.edge_progress < self.edge_progress:
						# Other vehicle is ahead on same path (reverse direction)
						distance = (self.edge_progress - other_vehicle.edge_progress) * edge_length
						if distance < self.minimum_distance:
							return True
					else:
						# Handle edge cases - vehicle at exact start or end of an edge
						# Check if vehicles are extremely close to each other at nodes
						if abs(self.edge_progress - other_vehicle.edge_progress) < 0.1:
							return True
			
			# Check if there's a vehicle ahead on same edge moving in same direction
			if (hasattr(other_vehicle, 'current_edge') and
					other_vehicle.current_edge == self.current_edge):
				
				# CRITICAL BUGFIX: Skip this check if other vehicle has lower priority than us
				# This allows higher priority vehicles to move first from the same node
				if other_vehicle.id > self.id and abs(other_vehicle.edge_progress - self.edge_progress) < 0.2:
					continue
				
				# Determine other vehicle's direction
				other_going_forward = (abs(other_vehicle.current_node[0] - x1) < 0.1 and
				                       abs(other_vehicle.current_node[1] - y1) < 0.1)
				
				# If traveling in same direction, check if ahead of us
				if going_forward == other_going_forward:
					# Fix this calculation
					edge_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
					
					# If same direction, check progress
					if going_forward and other_vehicle.edge_progress > self.edge_progress:
						# Other vehicle is ahead on same path
						distance = (other_vehicle.edge_progress - self.edge_progress) * edge_length
						if distance < self.minimum_distance:
							return True
					elif not going_forward and other_vehicle.edge_progress < self.edge_progress:
						# Other vehicle is ahead on same path (reverse direction)
						distance = (self.edge_progress - other_vehicle.edge_progress) * edge_length
						if distance < self.minimum_distance:
							return True
					# Add missing else branch to handle edge cases
					else:
						# Only consider vehicles with higher priority at the same position
						if other_vehicle.id < self.id and abs(self.edge_progress - other_vehicle.edge_progress) < 0.1:
							return True
		
		return False
	
	def _has_higher_priority_vehicle_at_node(self, system):
		"""
		Check if there's a vehicle with higher priority (lower ID) at the same node.
		Returns True if there's a vehicle with higher priority waiting at the same node.
		"""
		if not hasattr(self, 'current_node'):
			return False
		
		for other_vehicle in system.vehicles:
			if other_vehicle is self:
				continue
			
			# Check if other vehicle is at the same node
			if (hasattr(other_vehicle, 'current_node') and
					other_vehicle.current_node == self.current_node):
				
				# If other vehicle has a lower ID (higher priority) and is waiting
				if other_vehicle.id < self.id and other_vehicle.waiting:
					return True
		
		return False


class TrafficLight:
	def __init__(self, id, state="red", position=0):
		self.id = id
		self.state = state  # "red" or "green"
		self.position = position  # For a line, this is distance along line
		self.timer = 0
		self.cycle_duration = 15  # Time to switch between red and green
		
		# For 2D positions (used in square scenario)
		if isinstance(position, tuple) and len(position) == 2:
			self.x, self.y = position
		else:
			self.x, self.y = position, 0
	
	def update(self, dt):
		"""Update the traffic light's state based on the timer"""
		self.timer += dt
		if self.timer >= self.cycle_duration:
			self.timer = 0
			self.state = "green" if self.state == "red" else "red"


def run_simulation_for_optimization(green_durations, red_durations, delays, spawn_times=None, total_vehicles=None,
                                    system=None, vehicle_gap=10.0, traffic_rate=0.2, max_simulation_time=300):
	"""
	Run a simulation with the given parameters and return metrics.
	This function is used by the optimizer to evaluate different light timings.

	New feature: Can use vehicle_gap and traffic_rate for dynamic vehicle generation based on intervals

	Parameters:
	-----------
	green_durations : list
		Green light durations for each traffic light
	red_durations : list
		Red light durations for each traffic light
	delays : list
		Initial delays for each traffic light
	spawn_times : list, optional
		Times when vehicles will spawn (for time-based generation)
	total_vehicles : int, optional
		Total number of vehicles (for time-based generation)
	system : TrafficSystem, optional
		Existing system to base the simulation on (for road length, etc.)
	vehicle_gap : float, optional
		Minimum gap between vehicle spawns (seconds)
	traffic_rate : float or function, optional
		Vehicle arrival rate (per second), can be a constant or a function that takes current time and returns a rate
	max_simulation_time : float, optional
		Maximum simulation time (seconds)

	Returns:
	--------
	dict: Simulation metrics including:
		- exited_vehicles: Number of vehicles that completed the route
		- stops_per_light: List with number of stops at each light
		- simulation_time: Total simulation time
	"""
	
	# Create a temporary system for simulation
	temp_system = TrafficSystem()
	
	# Set up parameters for the simulation
	if system is not None:
		road_length = system.total_perimeter
		light_positions = [light.position for light in system.lights]
		vehicle_velocity = 0.4  # Default velocity
	else:
		road_length = 30
		light_positions = [5, 10, 15, 18, 25]  # Default positions
		vehicle_velocity = 0.4
	
	# Initialize the system
	temp_system.line_with_light(road_length, light_positions)
	
	# Configure traffic lights with the given parameters
	for i, light in enumerate(temp_system.lights):
		if i < len(green_durations) and i < len(red_durations) and i < len(delays):
			light.cycle_duration = green_durations[i] + red_durations[i]
			light.timer = delays[i] % light.cycle_duration
			# Start with green to maximize throughput at the beginning
			light.state = "green"
	
	# Initialize data structures to track vehicle information
	vehicle_stops = {}  # Track stops for each vehicle
	vehicles_spawned = 0
	vehicles_exited = 0
	stopped_at_light = {i: set() for i in range(len(temp_system.lights))}  # Track which vehicles stopped at each light
	
	# Run the simulation
	current_time = 0
	dt = 0.1  # Small time step for accurate simulation
	last_car_time = -vehicle_gap  # Last time a vehicle was spawned
	
	# Determine which vehicle generation method to use
	use_gap_based_spawning = spawn_times is None or total_vehicles is None
	
	if not use_gap_based_spawning:
		# If total_vehicles is not specified but spawn_times is provided
		if total_vehicles is None:
			total_vehicles = len(spawn_times)
	else:
		# When using gap-based generation, set a reasonable maximum number of vehicles
		if total_vehicles is None:
			total_vehicles = int(max_simulation_time / vehicle_gap) * 2
	
	# Continue until all vehicles have exited or maximum time reached
	while current_time < max_simulation_time:
		# Spawn vehicles
		if use_gap_based_spawning:
			# Dynamic vehicle generation based on gap
			if current_time - last_car_time >= vehicle_gap:
				# Get current traffic rate
				current_rate = traffic_rate
				if callable(traffic_rate):
					current_rate = traffic_rate(current_time)
				
				# Calculate probability of a vehicle arriving in this time step
				arrival_prob = current_rate * dt
				
				# Decide whether to spawn a vehicle based on probability
				if np.random.random() < arrival_prob and vehicles_spawned < total_vehicles:
					# Check if spawn area is clear
					spawn_area_clear = True
					for vehicle in temp_system.vehicles:
						if vehicle.position < 2.0:  # Check if any vehicle is in the first 2 units
							spawn_area_clear = False
							break
					
					if spawn_area_clear:
						# Create a new vehicle
						vehicle = Vehicle(id=vehicles_spawned + 1,
						                  position=0,
						                  length=1,
						                  velocity=vehicle_velocity)
						temp_system.add_vehicle(vehicle)
						vehicle_stops[vehicle.id] = 0
						vehicles_spawned += 1
						last_car_time = current_time
		else:
			# Time-based vehicle generation
			while vehicles_spawned < len(spawn_times) and current_time >= spawn_times[vehicles_spawned]:
				# Check if spawn area is clear
				spawn_area_clear = True
				for vehicle in temp_system.vehicles:
					if vehicle.position < 2.0:  # Check if any vehicle is in the first 2 units
						spawn_area_clear = False
						break
				
				if spawn_area_clear:
					# Create a new vehicle
					vehicle = Vehicle(id=vehicles_spawned + 1,
					                  position=0,
					                  length=1,
					                  velocity=vehicle_velocity)
					temp_system.add_vehicle(vehicle)
					vehicle_stops[vehicle.id] = 0
					vehicles_spawned += 1
		
		# Update all vehicles
		vehicles_to_remove = []
		for vehicle in temp_system.vehicles:
			# Check for stops at traffic lights
			for i, light in enumerate(temp_system.lights):
				# For line simulation, check if vehicle is stopped at this light
				if (vehicle.waiting or vehicle.stopped) and vehicle._is_at_light(light, temp_system):
					if vehicle.id not in stopped_at_light[i]:
						stopped_at_light[i].add(vehicle.id)
						vehicle_stops[vehicle.id] = vehicle_stops.get(vehicle.id, 0) + 1
			
			# Update vehicle position
			if vehicle.update(dt, temp_system):
				vehicles_to_remove.append(vehicle)
		
		# Remove vehicles that have reached the end
		for vehicle in vehicles_to_remove:
			if vehicle in temp_system.vehicles:
				temp_system.vehicles.remove(vehicle)
				vehicles_exited += 1
		
		# Update traffic lights with custom cycle durations
		for i, light in enumerate(temp_system.lights):
			if i < len(green_durations) and i < len(red_durations):
				light.timer += dt
				if light.state == "green" and light.timer >= green_durations[i]:
					light.state = "red"
					light.timer = 0
				elif light.state == "red" and light.timer >= red_durations[i]:
					light.state = "green"
					light.timer = 0
		
		# Increment time
		current_time += dt
		
		# Break early if all vehicles have been spawned and exited
		if (not use_gap_based_spawning and vehicles_spawned >= total_vehicles and vehicles_exited >= total_vehicles) or \
				(use_gap_based_spawning and vehicles_spawned >= total_vehicles and vehicles_exited >= vehicles_spawned):
			break
	
	# Calculate final metrics
	stops_per_light = [len(stopped_at_light[i]) for i in range(len(temp_system.lights))]
	total_stops = sum(vehicle_stops.values())
	
	return {
		'exited_vehicles': vehicles_exited,
		'stops_per_light': stops_per_light,
		'total_stops': total_stops,
		'vehicles_spawned': vehicles_spawned,
		'avg_stops_per_vehicle': total_stops / vehicles_spawned if vehicles_spawned > 0 else 0,
		'simulation_time': current_time,
		'completion_rate': vehicles_exited / vehicles_spawned if vehicles_spawned > 0 else 0
	}


class TrafficLightOptimizer:
	"""
	A class for optimizing traffic light timing parameters based on vehicle spawn patterns.
	Can be used with any traffic simulation configuration.
	"""
	
	def __init__(self,
	             light_positions,
	             velocity=14,
	             min_green=20,
	             max_green=40,
	             min_red=6.67,
	             max_red=13.33):
		"""
		Initialize the optimizer with simulation parameters.

		Parameters:
		-----------
		light_positions : list
			Positions of traffic lights (e.g., [500, 800, 1500])
		velocity : float
			Vehicle velocity in distance units per second
		min_green : float
			Minimum green light duration (seconds)
		max_green : float
			Maximum green light duration (seconds)
		min_red : float
			Minimum red light duration (seconds)
		max_red : float
			Maximum red light duration (seconds)
		"""
		self.light_positions = light_positions
		self.velocity = velocity
		self.min_green = min_green
		self.max_green = max_green
		self.min_red = min_red
		self.max_red = max_red
		self.delays = self.calculate_delays()
	
	def calculate_delays(self):
		"""
		Calculate the travel delays between traffic lights based on positions and velocity.

		Returns:
		--------
		list: Delays for each traffic light
		"""
		delays = [self.light_positions[0] / self.velocity]  # First light delay
		
		# Calculate delays for subsequent lights
		for i in range(1, len(self.light_positions)):
			delay = (self.light_positions[i] - self.light_positions[i - 1] - 2) / self.velocity
			delays.append(delay)
		
		return delays
	
	def calculate_red_duration(self, green_duration):
		"""
		Calculate red light duration based on green duration.

		Parameters:
		-----------
		green_duration : float
			Duration of green light

		Returns:
		--------
		float: Duration of red light
		"""
		return max(self.min_red, min(self.max_red, green_duration / 3))
	
	def optimize(self, spawn_times=None, total_vehicles=None, simulation_function=None,
	             max_iter=20, pop_size=10, strategy='best1bin', vehicle_gap=10.0, traffic_rate=0.2):
		"""
		Optimize traffic light timing using differential evolution.

		Added support for gap-based simulation with vehicle_gap and traffic_rate parameters
		instead of spawn_times.

		Parameters:
		-----------
		spawn_times : list, optional
			Times when vehicles will spawn (for time-based generation)
		total_vehicles : int, optional
			Total number of vehicles (for time-based generation)
		simulation_function : function
			Function to run a simulation with given parameters. Should accept:
				- green_durations: list of green light durations
				- red_durations: list of red light durations
				- delays: list of delays for each traffic light
				- spawn_times: list of vehicle spawn times (optional)
				- total_vehicles: total number of vehicles (optional)
				- vehicle_gap: minimum gap between vehicle spawns (optional)
				- traffic_rate: vehicle arrival rate (optional)
			And return a dictionary with at least:
				- 'exited_vehicles': number of vehicles that exited the system
				- 'stops_per_light': list of stop counts at each light
		max_iter : int
			Maximum number of iterations for optimization
		pop_size : int
			Population size for differential evolution
		strategy : str
			Strategy for differential evolution
		vehicle_gap : float, optional
			Minimum gap between vehicle spawns (seconds)
		traffic_rate : float or function, optional
			Vehicle arrival rate (per second), can be a constant or a function
			that takes current time and returns a rate

		Returns:
		--------
		dict: Optimization results including:
			- 'green_durations': Optimal green light durations
			- 'red_durations': Corresponding red light durations
			- 'delays': Delays for each traffic light
			- 'objective_value': Final objective function value
		"""
		import time
		from scipy.optimize import differential_evolution
		
		# Set bounds for optimization (green durations for each light)
		bounds = [(self.min_green, self.max_green) for _ in range(len(self.light_positions))]
		
		# Define simulation mode
		use_gap_based_simulation = spawn_times is None or total_vehicles is None
		
		# Define the objective function
		def objective(green_durations):
			# Calculate red durations
			red_durations = [self.calculate_red_duration(g) for g in green_durations]
			
			# Run simulation with appropriate parameters based on simulation mode
			if use_gap_based_simulation:
				# Gap-based simulation
				metrics = simulation_function(
					green_durations=green_durations,
					red_durations=red_durations,
					delays=self.delays,
					vehicle_gap=vehicle_gap,
					traffic_rate=traffic_rate
				)
			else:
				# Time-based simulation
				metrics = simulation_function(
					green_durations=green_durations,
					red_durations=red_durations,
					delays=self.delays,
					spawn_times=spawn_times,
					total_vehicles=total_vehicles
				)
			
			# Objective: minimize remaining cars and stops
			if use_gap_based_simulation:
				total_vehicles_simulated = metrics['vehicles_spawned']
			else:
				total_vehicles_simulated = total_vehicles
			
			remaining_cars = total_vehicles_simulated - metrics['exited_vehicles']
			exit_penalty = remaining_cars / total_vehicles_simulated if total_vehicles_simulated > 0 else 0
			
			# Add penalty for stopped vehicles
			total_stops = sum(metrics.get('stops_per_light', [0] * len(self.light_positions)))
			avg_stops = total_stops / total_vehicles_simulated if total_vehicles_simulated > 0 else 0
			
			# Combined objective
			return 10 * avg_stops
		
		# Run differential evolution
		print("\nStarting traffic light optimization...")
		print(f"Optimizing for {len(self.light_positions)} traffic lights")
		start_time = time.time()
		
		result = differential_evolution(
			objective,
			bounds,
			maxiter=max_iter,
			popsize=pop_size,
			strategy=strategy,
			disp=True
		)
		
		optimization_time = time.time() - start_time
		print(f"Optimization completed in {optimization_time:.2f} seconds")
		
		# Extract optimal parameters
		green_durations = result.x
		red_durations = [self.calculate_red_duration(g) for g in green_durations]
		
		# Print results
		print("\nOptimization Results:")
		print(f"Objective function value: {result.fun}")
		for i in range(len(green_durations)):
			print(f"Light {i + 1}: Green = {green_durations[i]:.2f}s, Red = {red_durations[i]:.2f}s")
		for i in range(len(self.delays)):
			print(f"Delay {i + 1}: {self.delays[i]:.2f}s")
		
		# Run final simulation to get detailed metrics
		if use_gap_based_simulation:
			final_metrics = simulation_function(
				green_durations=green_durations,
				red_durations=red_durations,
				delays=self.delays,
				vehicle_gap=vehicle_gap,
				traffic_rate=traffic_rate
			)
			total_vehicles_final = final_metrics['vehicles_spawned']
		else:
			final_metrics = simulation_function(
				green_durations=green_durations,
				red_durations=red_durations,
				delays=self.delays,
				spawn_times=spawn_times,
				total_vehicles=total_vehicles
			)
			total_vehicles_final = total_vehicles
		
		exited_percentage = final_metrics[
			                    'exited_vehicles'] / total_vehicles_final * 100 if total_vehicles_final > 0 else 0
		total_stops = sum(final_metrics.get('stops_per_light', [0]))
		avg_stops = total_stops / total_vehicles_final if total_vehicles_final > 0 else 0
		
		print(f"\nDetailed Final Metrics:")
		print(
			f"Vehicles exited: {final_metrics['exited_vehicles']} of {total_vehicles_final} ({exited_percentage:.1f}%)")
		print(f"Total stops: {total_stops}")
		print(f"Average stops per vehicle: {avg_stops:.2f}")
		print(f"Time to complete: {final_metrics.get('simulation_time', 0):.2f}s")
		
		# Return optimization results with metrics
		return {
			'green_durations': green_durations,
			'red_durations': red_durations,
			'delays': self.delays,
			'objective_value': result.fun,
			'final_metrics': final_metrics
		}


# Example traffic rate function
def traffic_rate_example(current_time):
	"""
	An example function for generating dynamic traffic rates
	Varies traffic intensity with time to simulate morning and evening rush hours

	Parameters:
	-----------
	current_time : float
		Current simulation time (seconds)

	Returns:
	--------
	float: Current traffic rate (vehicles per second)
	"""
	import math
	# Base traffic rate
	base_rate = 0.1
	# 8-hour period, simulating daily traffic variations
	period = 8 * 60 * 60
	# Peak traffic rate
	peak_rate = 0.5
	
	# Use sine function to simulate traffic fluctuations
	# Two peaks in the 0-4 hour and 4-8 hour periods (morning and evening rush hours)
	time_in_period = current_time % period
	hour_in_period = time_in_period / 3600
	
	if hour_in_period < 4:
		# Morning rush hour (peak at 2 hours)
		return base_rate + peak_rate * math.sin((hour_in_period / 4) * math.pi)
	else:
		# Evening rush hour (peak at 6 hours)
		return base_rate + peak_rate * math.sin(((hour_in_period - 4) / 4) * math.pi)


# class TrafficLightOptimizer:
# 	"""
# 	A class for optimizing traffic light timing parameters based on vehicle spawn patterns.
# 	Can be used with any traffic simulation configuration.
# 	"""
#
# 	def __init__(self,
# 	             light_positions,
# 	             velocity=14,
# 	             min_green=20,
# 	             max_green=40,
# 	             min_red=6.67,
# 	             max_red=13.33):
# 		"""
# 		Initialize the optimizer with simulation parameters.

# 		Parameters:
# 		-----------
# 		light_positions : list
# 			Positions of traffic lights (e.g., [500, 800, 1500])
# 		velocity : float
# 			Vehicle velocity in distance units per second
# 		min_green : float
# 			Minimum green light duration (seconds)
# 		max_green : float
# 			Maximum green light duration (seconds)
# 		min_red : float
# 			Minimum red light duration (seconds)
# 		max_red : float
# 			Maximum red light duration (seconds)
# 		"""
# 		self.light_positions = light_positions
# 		self.velocity = velocity
# 		self.min_green = min_green
# 		self.max_green = max_green
# 		self.min_red = min_red
# 		self.max_red = max_red
# 		self.delays = self.calculate_delays()
#
# 	def calculate_delays(self):
# 		"""
# 		Calculate the travel delays between traffic lights based on positions and velocity.

# 		Returns:
# 		--------
# 		list: Delays for each traffic light
# 		"""
# 		delays = [self.light_positions[0] / self.velocity]  # First light delay
#
# 		# Calculate delays for subsequent lights
# 		for i in range(1, len(self.light_positions)):
# 			delay = (self.light_positions[i] - self.light_positions[i - 1] - 2) / self.velocity
# 			delays.append(delay)
#
# 		return delays
#
# 	def calculate_red_duration(self, green_duration):
# 		"""
# 		Calculate red light duration based on green duration.

# 		Parameters:
# 		-----------
# 		green_duration : float
# 			Duration of green light

# 		Returns:
# 		--------
# 		float: Duration of red light
# 		"""
# 		return max(self.min_red, min(self.max_red, green_duration / 3))
#
# 	def optimize(self, spawn_times, total_vehicles, simulation_function,
# 	             max_iter=20, pop_size=10, strategy='best1bin'):
# 		"""
# 		Optimize traffic light timing using differential evolution.

# 		Parameters:
# 		-----------
# 		spawn_times : list
# 			Times when vehicles were spawned into the system
# 		total_vehicles : int
# 			Total number of vehicles that entered the system
# 		simulation_function : function
# 			Function to run a simulation with given parameters. Should accept:
# 				- green_durations: list of green light durations
# 				- red_durations: list of red light durations
# 				- delays: list of delays for each traffic light
# 				- spawn_times: list of vehicle spawn times
# 				- total_vehicles: total number of vehicles
# 			And return a dictionary with at least:
# 				- 'exited_vehicles': number of vehicles that exited the system
# 				- 'stops_per_light': list of stop counts at each light
# 		max_iter : int
# 			Maximum number of iterations for optimization
# 		pop_size : int
# 			Population size for differential evolution
# 		strategy : str
# 			Strategy for differential evolution

# 		Returns:
# 		--------
# 		dict: Optimization results including:
# 			- 'green_durations': Optimal green light durations
# 			- 'red_durations': Corresponding red light durations
# 			- 'delays': Delays for each traffic light
# 			- 'objective_value': Final objective function value
# 		"""
# 		# Set bounds for optimization (green durations for each light)
# 		bounds = [(self.min_green, self.max_green) for _ in range(len(self.light_positions))]
#
# 		# Define the objective function
# 		def objective(green_durations):
# 			# Calculate red durations
# 			red_durations = [self.calculate_red_duration(g) for g in green_durations]
#
# 			# Run simulation with these parameters
# 			metrics = simulation_function(
# 				green_durations=green_durations,
# 				red_durations=red_durations,
# 				delays=self.delays,
# 				spawn_times=spawn_times,
# 				total_vehicles=total_vehicles
# 			)
#
# 			# Objective: minimize remaining cars and stops
# 			remaining_cars = total_vehicles - metrics['exited_vehicles']
# 			exit_penalty = remaining_cars / total_vehicles if total_vehicles > 0 else 0
#
# 			# Add penalty for stopped vehicles
# 			total_stops = sum(metrics.get('stops_per_light', [0] * len(self.light_positions)))
# 			avg_stops = total_stops / total_vehicles if total_vehicles > 0 else 0
#
# 			# Combined objective
# 			return 10 * avg_stops
#
# 		# Run differential evolution
# 		print("\nStarting traffic light optimization...")
# 		print(f"Optimizing for {len(self.light_positions)} traffic lights")
# 		start_time = time.time()
#
# 		result = differential_evolution(
# 			objective,
# 			bounds,
# 			maxiter=max_iter,
# 			popsize=pop_size,
# 			strategy=strategy,
# 			disp=True
# 		)
#
# 		optimization_time = time.time() - start_time
# 		print(f"Optimization completed in {optimization_time:.2f} seconds")
#
# 		# Extract optimal parameters
# 		green_durations = result.x
# 		red_durations = [self.calculate_red_duration(g) for g in green_durations]
#
# 		# Print results
# 		print("\nOptimization Results:")
# 		print(f"Objective function value: {result.fun}")
# 		for i in range(len(green_durations)):
# 			print(f"Light {i + 1}: Green = {green_durations[i]:.2f}s, Red = {red_durations[i]:.2f}s")
# 		for i in range(len(self.delays)):
# 			print(f"Delay {i + 1}: {self.delays[i]:.2f}s")
#
# 		# Run final simulation to get detailed metrics
# 		final_metrics = simulation_function(
# 			green_durations=green_durations,
# 			red_durations=red_durations,
# 			delays=self.delays,
# 			spawn_times=spawn_times,
# 			total_vehicles=total_vehicles
# 		)
#
# 		exited_percentage = final_metrics['exited_vehicles'] / total_vehicles * 100 if total_vehicles > 0 else 0
# 		total_stops = sum(final_metrics.get('stops_per_light', [0]))
# 		avg_stops = total_stops / total_vehicles if total_vehicles > 0 else 0
#
# 		print(f"\nDetailed Final Metrics:")
# 		print(f"Vehicles exited: {final_metrics['exited_vehicles']} of {total_vehicles} ({exited_percentage:.1f}%)")
# 		print(f"Total stops: {total_stops}")
# 		print(f"Average stops per vehicle: {avg_stops:.2f}")
# 		print(f"Time to complete: {final_metrics.get('simulation_time', 0):.2f}s")
#
# 		# Return optimization results with metrics
# 		return {
# 			'green_durations': green_durations,
# 			'red_durations': red_durations,
# 			'delays': self.delays,
# 			'objective_value': result.fun,
# 			'final_metrics': final_metrics
# 		}


# def run_simulation_for_optimization(green_durations, red_durations, delays, spawn_times, total_vehicles, system=None):
# 	"""
# 	Run a simulation with the given parameters and return metrics.
# 	This function is used by the optimizer to evaluate different light timings.

# 	Parameters:
# 	-----------
# 	green_durations : list
# 		Green light durations for each traffic light
# 	red_durations : list
# 		Red light durations for each traffic light
# 	delays : list
# 		Initial delays for each traffic light
# 	spawn_times : list
# 		Times when vehicles will spawn
# 	total_vehicles : int
# 		Total number of vehicles
# 	system : TrafficSystem, optional
# 		Existing system to base the simulation on (for road length, etc.)

# 	Returns:
# 	--------
# 	dict: Simulation metrics including:
# 		- exited_vehicles: Number of vehicles that completed the route
# 		- stops_per_light: List with number of stops at each light
# 		- simulation_time: Total simulation time
# 	"""
# 	# Create a temporary system for simulation
# 	temp_system = TrafficSystem()
#
# 	# Set up parameters for the simulation
# 	if system is not None:
# 		road_length = system.total_perimeter
# 		light_positions = [light.position for light in system.lights]
# 		vehicle_velocity = 0.4  # Default velocity
# 	else:
# 		road_length = 30
# 		light_positions = [5, 10, 15, 18, 25]  # Default positions
# 		vehicle_velocity = 0.4
#
# 	# Initialize the system
# 	temp_system.line_with_light(road_length, light_positions)
#
# 	# Configure traffic lights with the given parameters
# 	for i, light in enumerate(temp_system.lights):
# 		if i < len(green_durations) and i < len(red_durations) and i < len(delays):
# 			light.cycle_duration = green_durations[i] + red_durations[i]
# 			light.timer = delays[i] % light.cycle_duration
# 			# Start with green to maximize throughput at the beginning
# 			light.state = "green"
#
# 	# Initialize data structures to track vehicle information
# 	vehicle_stops = {}  # Track stops for each vehicle
# 	vehicles_spawned = 0
# 	vehicles_exited = 0
# 	stopped_at_light = {i: set() for i in range(len(temp_system.lights))}  # Track which vehicles stopped at each light
#
# 	# Run the simulation
# 	current_time = 0
# 	dt = 0.1  # Small time step for accurate simulation
# 	max_simulation_time = 300  # Maximum simulation time (prevent infinite loops)
#
# 	# Continue until all vehicles have exited or maximum time reached
# 	while current_time < max_simulation_time:
# 		# Spawn vehicles according to spawn times
# 		while vehicles_spawned < len(spawn_times) and current_time >= spawn_times[vehicles_spawned]:
# 			# Check if spawn area is clear
# 			spawn_area_clear = True
# 			for vehicle in temp_system.vehicles:
# 				if vehicle.position < 2.0:  # Check if any vehicle is in the first 2 units
# 					spawn_area_clear = False
# 					break
#
# 			if spawn_area_clear:
# 				# Create a new vehicle
# 				vehicle = Vehicle(id=vehicles_spawned + 1,
# 				                  position=0,
# 				                  length=1,
# 				                  velocity=vehicle_velocity)
# 				temp_system.add_vehicle(vehicle)
# 				vehicle_stops[vehicle.id] = 0
# 				vehicles_spawned += 1
#
# 		# Update all vehicles
# 		vehicles_to_remove = []
# 		for vehicle in temp_system.vehicles:
# 			# Check for stops at traffic lights
# 			for i, light in enumerate(temp_system.lights):
# 				# For line simulation, check if vehicle is stopped at this light
# 				if (vehicle.waiting or vehicle.stopped) and vehicle._is_at_light(light, temp_system):
# 					if vehicle.id not in stopped_at_light[i]:
# 						stopped_at_light[i].add(vehicle.id)
# 						vehicle_stops[vehicle.id] = vehicle_stops.get(vehicle.id, 0) + 1
#
# 			# Update vehicle position
# 			if vehicle.update(dt, temp_system):
# 				vehicles_to_remove.append(vehicle)
#
# 		# Remove vehicles that have reached the end
# 		for vehicle in vehicles_to_remove:
# 			if vehicle in temp_system.vehicles:
# 				temp_system.vehicles.remove(vehicle)
# 				vehicles_exited += 1
#
# 		# Update traffic lights with custom cycle durations
# 		for i, light in enumerate(temp_system.lights):
# 			if i < len(green_durations) and i < len(red_durations):
# 				light.timer += dt
# 				if light.state == "green" and light.timer >= green_durations[i]:
# 					light.state = "red"
# 					light.timer = 0
# 				elif light.state == "red" and light.timer >= red_durations[i]:
# 					light.state = "green"
# 					light.timer = 0
#
# 		# Increment time
# 		current_time += dt
#
# 		# Break early if all vehicles have been spawned and exited
# 		if vehicles_spawned >= total_vehicles and vehicles_exited >= total_vehicles:
# 			break
#
# 	# Calculate final metrics
# 	stops_per_light = [len(stopped_at_light[i]) for i in range(len(temp_system.lights))]
# 	total_stops = sum(vehicle_stops.values())
#
# 	return {
# 		'exited_vehicles': vehicles_exited,
# 		'stops_per_light': stops_per_light,
# 		'total_stops': total_stops,
# 		'avg_stops_per_vehicle': total_stops / total_vehicles if total_vehicles > 0 else 0,
# 		'simulation_time': current_time,
# 		'completion_rate': vehicles_exited / total_vehicles if total_vehicles > 0 else 0
# 	}


class TrafficLightController:
	def __init__(self, system):
		self.system = system
		self.control_modes = {
			"fixed": self.fixed_timing_control,
			"green_wave": self.green_wave_control,
			"optimization": self.optimization_control,
		}
		self.current_mode = "fixed"
		self.cycle_duration_base = 15  # default cycle duration
		self.min_green_time = 5  # minimum green time for a light
		self.max_red_time = 30  # maximum time a light can stay red
		self.optimization_window = 60  # look-ahead time window for optimization
		self.last_optimization = 0  # when we last ran the optimization
		self.optimization_interval = 5  # how often to re-optimize
		self.stop_start_count = 0  # counter for stop-starts
		self.light_schedule = {}  # planned light changes from optimization
	
	def set_control_mode(self, mode):
		"""Set the traffic light control mode"""
		if mode in self.control_modes:
			self.current_mode = mode
			print(f"Traffic light control mode set to: {mode}")
			self._reset_light_timers()
			return True
		return False
	
	def set_cycle_duration(self, duration):
		"""Set the base cycle duration for all traffic lights"""
		if duration > 0:
			self.cycle_duration_base = float(duration)
			for light in self.system.lights:
				light.cycle_duration = self.cycle_duration_base
			print(f"Cycle duration set to: {duration} seconds")
			return True
		return False
	
	def update(self, dt):
		"""Update traffic lights based on the selected control mode"""
		control_func = self.control_modes.get(self.current_mode, self.fixed_timing_control)
		control_func(dt)
	
	def fixed_timing_control(self, dt):
		"""Simple fixed timing control (default behavior)"""
		for light in self.system.lights:
			light.update(dt)
	
	def green_wave_control(self, dt):
		"""Create a green wave effect where only one light is green at a time"""
		sorted_light_ids = [light.id for light in self.system.lights]
		
		# Calculate total cycle time based on number of lights
		total_cycle_time = self.cycle_duration_base * len(sorted_light_ids)
		
		# Determine which light should be green based on current time
		current_phase = (self.system.time % total_cycle_time) // self.cycle_duration_base
		active_light_index = int(current_phase) % len(sorted_light_ids)
		
		# Update each light - only one is green at a time
		for i, light in enumerate(self.system.lights):
			light.state = "green" if i == active_light_index else "red"
			light.timer = (self.system.time % self.cycle_duration_base) if i == active_light_index else 0
	
	def optimization_control(self, dt):
		"""
		Control traffic lights using optimization to minimize vehicle stops.
		Optimizes on a periodic basis rather than every time step.
		"""
		# Only re-optimize periodically to save computation
		if self.system.time - self.last_optimization >= self.optimization_interval:
			self._optimize_light_timings()
			self.last_optimization = self.system.time
		
		# Apply the pre-computed schedule
		current_time = self.system.time
		for light in self.system.lights:
			if light.id in self.light_schedule:
				# Check if we need to change the light state based on schedule
				scheduled_changes = self.light_schedule[light.id]
				while scheduled_changes and scheduled_changes[0][0] <= current_time:
					change_time, new_state = scheduled_changes.pop(0)
					light.state = new_state
					light.timer = current_time - change_time
	
	def _optimize_light_timings(self):
		"""
		Optimize traffic light timings to minimize vehicle stops.
		Uses a prediction of vehicle movements to schedule optimal green times.
		"""
		# Clear previous schedule
		self.light_schedule = {light.id: [] for light in self.system.lights}
		
		# Predict vehicle positions over the optimization window
		predicted_arrivals = self._predict_vehicle_arrivals()
		
		# Sort lights by position (for line) or ID (for other shapes)
		lights = sorted(self.system.lights,
		                key=lambda l: l.position if self.system.shape in ["line", "line_with_light"] else l.id)
		
		# Greedy optimization approach
		for light in lights:
			# Get predicted vehicle arrivals at this light
			light_arrivals = predicted_arrivals.get(light.id, [])
			if not light_arrivals:
				continue
			
			# Group arrivals into clusters that could cross on the same green
			clusters = self._cluster_arrivals(light_arrivals)
			
			# Schedule green windows for each significant cluster
			current_time = self.system.time
			state = light.state
			
			for cluster_start, cluster_end in clusters:
				# Calculate when to turn green (account for min_green_time constraints)
				green_start = max(current_time, cluster_start - 1)  # Turn green slightly before arrival
				green_duration = min(cluster_end - cluster_start + 2, self.cycle_duration_base)
				green_duration = max(green_duration, self.min_green_time)
				
				# Don't violate max_red_time constraint
				if state == "red" and green_start - current_time > self.max_red_time:
					green_start = current_time + self.max_red_time
				
				# Schedule the changes
				if state != "green":
					self.light_schedule[light.id].append((green_start, "green"))
					state = "green"
				
				# Schedule turn back to red after green period
				red_time = green_start + green_duration
				self.light_schedule[light.id].append((red_time, "red"))
				state = "red"
				current_time = red_time
	
	def _predict_vehicle_arrivals(self):
		"""
		Predict when vehicles will arrive at each light within the optimization window.
		Returns a dict mapping light IDs to lists of arrival times.
		"""
		arrivals = {light.id: [] for light in self.system.lights}
		
		for vehicle in self.system.vehicles:
			if vehicle.waiting or vehicle.stopped:
				continue  # Skip vehicles that are already stopped
			
			# For each light, predict when this vehicle will reach it
			for light in self.system.lights:
				light_pos = self._get_light_position(light)
				
				# If vehicle is before the light in its path
				if self._is_vehicle_approaching_light(vehicle, light):
					# Calculate arrival time based on position and velocity
					distance = self._calculate_distance_to_light(vehicle, light)
					if vehicle.velocity > 0:
						arrival_time = self.system.time + (distance / vehicle.velocity)
						
						# Only consider arrivals within our optimization window
						if arrival_time <= self.system.time + self.optimization_window:
							arrivals[light.id].append(arrival_time)
		
		return arrivals
	
	def _cluster_arrivals(self, arrival_times, max_gap=3):
		"""
		Group arrival times into clusters where vehicles arrive close together.
		Returns list of (start_time, end_time) tuples for each cluster.
		"""
		if not arrival_times:
			return []
		
		sorted_arrivals = sorted(arrival_times)
		clusters = []
		cluster_start = sorted_arrivals[0]
		cluster_end = sorted_arrivals[0]
		
		for time in sorted_arrivals[1:]:
			if time - cluster_end > max_gap:
				# Start a new cluster
				clusters.append((cluster_start, cluster_end))
				cluster_start = time
			cluster_end = time
		
		# Add the last cluster
		clusters.append((cluster_start, cluster_end))
		
		return clusters
	
	def _is_vehicle_approaching_light(self, vehicle, light):
		"""Determine if a vehicle is approaching a light."""
		if self.system.shape in ["line", "line_with_light"]:
			return vehicle.position < light.position
		
		# For square, more complex logic is needed based on the path
		# This is a simplification for the square case:
		vehicle_edge, _ = vehicle._get_current_edge_position(self.system)
		light_edge = -1
		
		# Find which edge the light is on
		for i, edge in enumerate(self.system.edges):
			x1, y1, x2, y2 = edge
			if light.x == x1 and light.y == y1:
				light_edge = i
				break
		
		# Vehicle is approaching if it's on the previous edge or early on the same edge
		if light_edge == -1:
			return False
		
		if vehicle_edge == (light_edge - 1) % len(self.system.edges):
			return True
		if vehicle_edge == light_edge:
			# On same edge, check if vehicle is before light
			if self.system.shape == "square":
				# Complex logic for square would go here
				# This is simplified
				return True
		
		return False
	
	def _calculate_distance_to_light(self, vehicle, light):
		"""Calculate the distance from vehicle to light along the path."""
		if self.system.shape in ["line", "line_with_light"]:
			return max(0, light.position - vehicle.position)
		
		# For square, would need to calculate distance along perimeter
		# This is a simplification
		return 5  # Placeholder
	
	def _get_light_position(self, light):
		"""Get the position of a light along the path."""
		if self.system.shape in ["line", "line_with_light"]:
			return light.position
		
		# For square, the position is the perimeter distance
		# This would need more calculation for full implementation
		return 0  # Placeholder
	
	def _reset_light_timers(self):
		"""Reset all traffic light timers"""
		for light in self.system.lights:
			light.timer = 0