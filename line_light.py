import pygame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from sim_line import TrafficSystem, Vehicle, TrafficLight, TrafficLightController, TrafficLightOptimizer
from metrics import Metrics
import numpy as np
import random
# Import the optimizer
# from traffic_optimization import TrafficLightOptimizer, detect_stops_at_traffic_lights
import time

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 500
BG_COLOR = (240, 240, 240)
ROAD_COLOR = (80, 80, 80)
VEHICLE_COLOR = (0, 0, 255)  # Blue color for all vehicles
RED_LIGHT_COLOR = (255, 0, 0)
GREEN_LIGHT_COLOR = (0, 255, 0)
LIGHT_RADIUS = 8
VEHICLE_SIZE = 10
FONT_SIZE = 24
BUTTON_COLOR = (100, 200, 100)
BUTTON_HOVER_COLOR = (120, 220, 120)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Traffic Simulation")
font = pygame.font.SysFont(None, FONT_SIZE)
light_font = pygame.font.SysFont(None, 20)  # Smaller font for light IDs


# Button class for UI interactions
class Button:
	def __init__(self, x, y, width, height, text, callback):
		self.rect = pygame.Rect(x, y, width, height)
		self.text = text
		self.callback = callback
		self.is_hovered = False
	
	def draw(self, screen):
		color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
		pygame.draw.rect(screen, color, self.rect)
		pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)  # Border
		text_surf = font.render(self.text, True, BUTTON_TEXT_COLOR)
		text_rect = text_surf.get_rect(center=self.rect.center)
		screen.blit(text_surf, text_rect)
	
	def check_hover(self, mouse_pos):
		self.is_hovered = self.rect.collidepoint(mouse_pos)
		return self.is_hovered
	
	def handle_event(self, event):
		if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
			self.callback()
			return True
		return False


def draw_line_simulation(screen, system):
	"""Draw a line simulation"""
	margin_x, margin_y = 50, SCREEN_HEIGHT // 2
	road_length = system.total_perimeter
	scale_factor = (SCREEN_WIDTH - 2 * margin_x) / road_length
	
	# Draw the road
	road_width = 20
	pygame.draw.rect(screen, ROAD_COLOR, (margin_x, margin_y - road_width // 2,
	                                      int(road_length * scale_factor), road_width))
	
	# Draw traffic lights
	for light in system.lights:
		light_x = margin_x + int(light.position * scale_factor)
		light_y = margin_y
		color = RED_LIGHT_COLOR if light.state == "red" else GREEN_LIGHT_COLOR
		# Debug print to see the light state
		# print(f"Drawing light {light.id} with state {light.state}")
		pygame.draw.circle(screen, color, (light_x, light_y), LIGHT_RADIUS)
		light_id_text = light_font.render(str(light.id), True, (0, 0, 0))
		light_id_rect = light_id_text.get_rect(center=(light_x, light_y - 15))
		screen.blit(light_id_text, light_id_rect)
	
	# Draw vehicles
	for vehicle in system.vehicles:
		vehicle_x = margin_x + int(vehicle.position * scale_factor)
		vehicle_y = margin_y
		vehicle_width = max(int(vehicle.length * scale_factor), VEHICLE_SIZE)
		pygame.draw.rect(screen, VEHICLE_COLOR,
		                 (vehicle_x - vehicle_width // 2, vehicle_y - VEHICLE_SIZE // 2,
		                  vehicle_width, VEHICLE_SIZE))
		id_text = font.render(str(vehicle.id), True, (255, 255, 255))
		id_rect = id_text.get_rect(center=(vehicle_x, vehicle_y))
		screen.blit(id_text, id_rect)


def draw_square_simulation(screen, system):
	"""Draw a square simulation"""
	side_length = system.edges[0][2]  # First edge's x2 is the side length
	scale_factor = min((SCREEN_WIDTH - 100) / side_length,
	                   (SCREEN_HEIGHT - 100) / side_length)
	
	# Screen margins to center the square
	margin_x = (SCREEN_WIDTH - side_length * scale_factor) // 2
	margin_y = (SCREEN_HEIGHT - side_length * scale_factor) // 2
	
	# Draw the road (square)
	road_width = 20
	for edge in system.edges:
		x1, y1, x2, y2 = edge
		start_pos = (margin_x + int(x1 * scale_factor), margin_y + int(y1 * scale_factor))
		end_pos = (margin_x + int(x2 * scale_factor), margin_y + int(y2 * scale_factor))
		pygame.draw.line(screen, ROAD_COLOR, start_pos, end_pos, road_width)
	
	# Draw traffic lights
	for light in system.lights:
		light_x = margin_x + int(light.x * scale_factor)
		light_y = margin_y + int(light.y * scale_factor)
		color = RED_LIGHT_COLOR if light.state == "red" else GREEN_LIGHT_COLOR
		pygame.draw.circle(screen, color, (light_x, light_y), LIGHT_RADIUS)
		light_id_text = light_font.render(str(light.id), True, (0, 0, 0))
		light_id_rect = light_id_text.get_rect(center=(light_x, light_y - 15))
		screen.blit(light_id_text, light_id_rect)
	
	# Draw vehicles
	for vehicle in system.vehicles:
		x, y = vehicle.get_coordinates(system)
		vehicle_x = margin_x + int(x * scale_factor)
		vehicle_y = margin_y + int(y * scale_factor)
		pygame.draw.circle(screen, VEHICLE_COLOR, (vehicle_x, vehicle_y), VEHICLE_SIZE // 2)
		id_text = font.render(str(vehicle.id), True, (0, 0, 0))
		id_rect = id_text.get_rect(center=(vehicle_x, vehicle_y - 15))
		screen.blit(id_text, id_rect)


def draw_grid_simulation(screen, system):
	"""Draw a grid network simulation with labeled nodes and edges"""
	# Calculate scaling to fit on screen
	max_x = max([max(edge[0], edge[2]) for edge in system.edges])
	max_y = max([max(edge[1], edge[3]) for edge in system.edges])
	
	scale_factor = min((SCREEN_WIDTH - 100) / max_x,
	                   (SCREEN_HEIGHT - 100) / max_y)
	
	# Screen margins to center the grid
	margin_x = (SCREEN_WIDTH - max_x * scale_factor) // 2
	margin_y = (SCREEN_HEIGHT - max_y * scale_factor) // 2
	
	# Collect all unique points to identify intersections
	intersection_points = set()
	for edge in system.edges:
		x1, y1, x2, y2 = edge
		intersection_points.add((x1, y1))
		intersection_points.add((x2, y2))
	
	# Draw the roads
	road_width = 15
	for i, edge in enumerate(system.edges):
		x1, y1, x2, y2 = edge
		start_pos = (margin_x + int(x1 * scale_factor), margin_y + int(y1 * scale_factor))
		end_pos = (margin_x + int(x2 * scale_factor), margin_y + int(y2 * scale_factor))
		pygame.draw.line(screen, ROAD_COLOR, start_pos, end_pos, road_width)
		
		# Label each edge with ID
		mid_x = margin_x + int((x1 + x2) / 2 * scale_factor)
		mid_y = margin_y + int((y1 + y2) / 2 * scale_factor)
		edge_label = light_font.render(f"E{i}", True, (255, 255, 255))
		edge_rect = edge_label.get_rect(center=(mid_x, mid_y))
		screen.blit(edge_label, edge_rect)
	
	# Draw intersections with coordinate labels
	node_font = pygame.font.SysFont(None, 16)
	for node_id, (x, y) in enumerate(sorted(intersection_points)):
		node_x = margin_x + int(x * scale_factor)
		node_y = margin_y + int(y * scale_factor)
		
		# Mark the node/intersection
		pygame.draw.circle(screen, (100, 100, 100), (node_x, node_y), 5)
		
		# Label with coordinates
		node_label = f"N{node_id}:({int(x)},{int(y)})"
		label_text = node_font.render(node_label, True, (0, 0, 0))
		label_rect = label_text.get_rect(center=(node_x, node_y - 15))
		screen.blit(label_text, label_rect)
	
	# Draw traffic lights
	for light in system.lights:
		light_x = margin_x + int(light.x * scale_factor)
		light_y = margin_y + int(light.y * scale_factor)
		color = RED_LIGHT_COLOR if light.state == "red" else GREEN_LIGHT_COLOR
		pygame.draw.circle(screen, color, (light_x, light_y), LIGHT_RADIUS)
	
	# Draw destinations with more visibility
	for vehicle in system.vehicles:
		if hasattr(vehicle, 'destination_node') and vehicle.destination_node is not None:
			dest_x, dest_y = vehicle.destination_node
			dest_screen_x = margin_x + int(dest_x * scale_factor)
			dest_screen_y = margin_y + int(dest_y * scale_factor)
			
			# Mark destination with more visible indicator
			pygame.draw.circle(screen, (255, 0, 0),
			                   (dest_screen_x, dest_screen_y), 5, 2)
	
	# First draw all Euclidean distance lines for all vehicles
	for vehicle in system.vehicles:
		if system.shape == "grid":
			x, y = vehicle._get_current_position(system)
		else:
			x, y = vehicle.get_coordinates(system)
		
		vehicle_x = margin_x + int(x * scale_factor)
		vehicle_y = margin_y + int(y * scale_factor)
		
		# Draw the Euclidean distance line to destination with increased visibility
		if hasattr(vehicle, 'destination_node') and vehicle.destination_node is not None:
			dest_x, dest_y = vehicle.destination_node
			dest_screen_x = margin_x + int(dest_x * scale_factor)
			dest_screen_y = margin_y + int(dest_y * scale_factor)
			
			# Draw straight-line path to destination (Euclidean distance) - THICKER and BRIGHTER
			pygame.draw.line(screen, (255, 0, 0),
			                 (vehicle_x, vehicle_y),
			                 (dest_screen_x, dest_screen_y), 1)  # Increased thickness to 2
	
	# Now draw the direction arrows and vehicles on top
	for vehicle in system.vehicles:
		if system.shape == "grid":
			x, y = vehicle._get_current_position(system)
		else:
			x, y = vehicle.get_coordinates(system)
		
		vehicle_x = margin_x + int(x * scale_factor)
		vehicle_y = margin_y + int(y * scale_factor)
		
		# Draw direction arrows with increased visibility
		if hasattr(vehicle, 'destination_node') and vehicle.destination_node is not None:
			dest_x, dest_y = vehicle.destination_node
			
			# Calculate direction vector
			dx = dest_x - x
			dy = dest_y - y
			
			# Normalize the direction vector
			length = np.sqrt(dx * dx + dy * dy)
			if length > 0:
				dx, dy = dx / length, dy / length
				
				# Calculate arrow position (bigger offset from vehicle)
				arrow_length = 20  # Increased from 15
				arrow_pos_x = vehicle_x + dx * 12
				arrow_pos_y = vehicle_y + dy * 12
				
				# Calculate arrowhead points
				arrow_end_x = arrow_pos_x + dx * arrow_length
				arrow_end_y = arrow_pos_y + dy * arrow_length
				
				# Draw arrow line - THICKER
				pygame.draw.line(screen, (0, 200, 0),  # Brighter green
				                 (arrow_pos_x, arrow_pos_y),
				                 (arrow_end_x, arrow_end_y), 3)  # Increased thickness
				
				# Draw arrowhead - LARGER
				arrow_size = 8  # Increased from 6
				angle = np.arctan2(dy, dx)
				pygame.draw.polygon(screen, (0, 200, 0), [  # Brighter green
					(arrow_end_x, arrow_end_y),
					(arrow_end_x - arrow_size * np.cos(angle - np.pi / 6),
					 arrow_end_y - arrow_size * np.sin(angle - np.pi / 6)),
					(arrow_end_x - arrow_size * np.cos(angle + np.pi / 6),
					 arrow_end_y - arrow_size * np.sin(angle + np.pi / 6))
				])
		
		# Draw vehicle with ID
		pygame.draw.circle(screen, VEHICLE_COLOR, (vehicle_x, vehicle_y), VEHICLE_SIZE // 2)
		id_text = font.render(str(vehicle.id), True, (255, 255, 255))
		id_rect = id_text.get_rect(center=(vehicle_x, vehicle_y))
		screen.blit(id_text, id_rect)
	
	# Add destination information panel
	display_vehicle_destinations(screen, system)


def draw_simulation_info(screen, system, controller=None, simulation_started=False, simulation_ended=False,
                         optimization_applied=False):
	"""Draw simulation information"""
	time_text = font.render(f"Time: {np.floor(system.time)}", True, (0, 0, 0))
	screen.blit(time_text, (10, 10))
	
	if simulation_ended:
		status_text = font.render("Simulation Completed!", True, (0, 150, 0))
		screen.blit(status_text, (10, 40))
	elif not simulation_started:
		status_text = font.render("Simulation Ready - Press Start", True, (150, 0, 0))
		screen.blit(status_text, (10, 40))
	else:
		status_text = font.render("Simulation Running", True, (0, 0, 150))
		screen.blit(status_text, (10, 40))
	
	if optimization_applied:
		mode_text = font.render("Control Mode: DE Optimized", True, (0, 0, 150))
	elif controller:
		mode_text = font.render(f"Control Mode: {controller.current_mode}", True, (0, 0, 0))
	else:
		mode_text = font.render("Control Mode: None", True, (0, 0, 0))
	screen.blit(mode_text, (10, 70))
	
	# if controller:
	# 	mode_text = font.render(f"Control Mode: {controller.current_mode}", True, (0, 0, 0))
	# 	screen.blit(mode_text, (10, 70))
	#
	y_offset = 130
	for vehicle in system.vehicles:
		vehicle_text = font.render(
			f"Vehicle {vehicle.id}: {'Stopped' if vehicle.waiting or vehicle.stopped else 'Moving'}",
			True, (0, 0, 0))
		screen.blit(vehicle_text, (10, y_offset))
		y_offset += 30


# ------ OPTIONAL ------ (show traffic light states)
# for light in system.lights:
#    light_text = font.render(f"Light {light.id}: {light.state}", True, (0, 0, 0))
#    screen.blit(light_text, (10, y_offset))
#    y_offset += 30

def draw_help_menu(screen):
	"""Draw help menu with controller instructions"""
	help_texts = [
		"Controls:",
		"SPACE - Pause/Resume",
		"H - Toggle Help",
		"1 - Fixed Timing Mode",
		"2 - Green Wave Mode",
		"3 - Optimization Mode",
		"R - Reset Simulation",
		"ESC - Quit"
	]
	
	help_y = 150
	for text in help_texts:
		help_surface = font.render(text, True, (0, 0, 0))
		screen.blit(help_surface, (SCREEN_WIDTH - 300, help_y))
		help_y += 30


def check_simulation_ended(system, simulation_type):
	"""Check if the simulation should end based on vehicle positions"""
	if simulation_type == "square":
		return False
	
	for vehicle in system.vehicles:
		if vehicle.position < system.total_perimeter:
			return False
	
	return True


def display_vehicle_destinations(screen, system):
	"""Display vehicle destination information in a panel"""
	info_font = pygame.font.SysFont(None, 18)
	panel_x = SCREEN_WIDTH - 200
	panel_y = 130
	
	# Panel title
	title_text = font.render("Vehicle Routes:", True, (0, 0, 0))
	screen.blit(title_text, (panel_x, panel_y))
	panel_y += 30
	
	# List each vehicle's destination
	for vehicle in system.vehicles:
		# Get current position
		curr_x, curr_y = vehicle._get_current_position(system) if system.shape == "grid" else vehicle.get_coordinates(
			system)
		
		# Try to get destination info
		dest_info = ""
		if hasattr(vehicle, 'destination_node') and vehicle.destination_node:
			dest_x, dest_y = vehicle.destination_node
			dest_info = f" → ({int(dest_x)},{int(dest_y)})"
		
		# Display vehicle info
		vehicle_info = f"V{vehicle.id}: ({int(curr_x)},{int(curr_y)}){dest_info}"
		info_text = info_font.render(vehicle_info, True, (0, 0, 0))
		screen.blit(info_text, (panel_x, panel_y))
		panel_y += 20


# Add this function to visualisation.py to plot the optimization results
def plot_optimization_results(green_durations, red_durations, light_positions, stops_at_lights=None,vehicle_trajectories=None):
	"""
	Create and display plots for traffic light timing visualization and stop frequency.

	Parameters:
	-----------
	green_durations: list
		List of green light durations for each traffic light
	red_durations: list
		List of red light durations for each traffic light
	light_positions: list
		List of positions for each traffic light
	stops_at_lights: list, optional
		Number of stops recorded at each traffic light
	"""
	print(f"Green durations: {green_durations}")
	print(f"Red durations: {red_durations}")
	print(f"Light positions: {light_positions}")
	
	assert len(green_durations) == len(light_positions), \
		f"Mismatch: {len(green_durations)} durations vs {len(light_positions)} lights"
	
	# Create a new figure with 2 subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
	
	# Plot 1: Traffic Light States Over Time
	max_time = 200  # Maximum simulation time to show
	time_steps = np.arange(0, max_time, 0.1)
	
	# For each traffic light, determine its state at each time step
	light_states = []
	for i in range(len(green_durations)):
		states = []
		cycle_time = green_durations[i] + red_durations[i]
		
		# Default starting in green state with some offset between lights
		initial_offset = i * 5  # Stagger the starts
		
		for t in time_steps:
			# Calculate where in the cycle this light is
			cycle_position = (t + initial_offset) % cycle_time
			# Green if in first part of cycle, red otherwise
			if cycle_position < green_durations[i]:
				states.append(1)  # Green
			else:
				states.append(0)  # Red
		light_states.append(states)
	
	# Plot each traffic light state
	colors = ['blue', 'orange', 'green', 'red', 'purple']
	for i, states in enumerate(light_states):
		if i < len(colors):
			ax1.step(time_steps, states, label=f'TL{i + 1}', where='post', color=colors[i])
	
	# Configure the plot
	ax1.set_ylim([-0.1, 1.1])
	ax1.set_xlim([0, max_time])
	ax1.set_yticks([0, 1])
	ax1.set_yticklabels(['Red', 'Green'])
	ax1.set_xlabel('Time (seconds)')
	ax1.set_ylabel('State (0=Red, 1=Green)')
	ax1.set_title('Traffic Light States Over Time')
	ax1.legend()
	ax1.grid(True)
	
	# Plot 2: Stop Frequency vs. Position
	max_position = max(light_positions) * 1.1
	position_bins = np.linspace(0, max_position, 100)
	print(f"Plotting bar chart...")
	print(f"Light Positions: {light_positions}")
	print(f"Stops at Lights: {stops_at_lights}")
	
	# If we have stop data, plot histogram of stops
	if stops_at_lights is not None:
		# First get the y-limit for the plot based on max stops
		max_stops = max(stops_at_lights) if stops_at_lights and max(stops_at_lights) > 0 else 1
		y_limit = max_stops * 1.2  # Give 20% margin at the top
		
		# Create a histogram-like visualization with better-colored bars
		bars = ax2.bar(light_positions, stops_at_lights, width=20, alpha=0.8, color='blue')
	
	# Add text labels on top of each bar showing the exact count
	for i, bar in enumerate(bars):
		height = bar.get_height()
		ax2.text(bar.get_x() + bar.get_width() / 2., height + max_stops * 0.05,
		         f'{stops_at_lights[i]}',
		         ha='center', va='bottom', fontweight='bold')
	
	# Set the y-limit to be consistent and show all data
	ax2.set_ylim(0, y_limit)
	
	# Mark traffic light positions with vertical lines
	for i, pos in enumerate(light_positions):
		ax2.axvline(x=pos, color='red', linestyle='--', label=f'TL{i + 1}' if i == 0 else "")
		ax2.text(pos, ax2.get_ylim()[1] * 0.9, f'TL{i + 1}', ha='center')
	
	# Add more descriptive title and other improvements
	ax2.set_title('Number of Vehicles Stopped at Each Traffic Light')
	ax2.set_ylabel('Number of Stopped Vehicles')
	ax2.grid(True, linestyle='--', alpha=0.7)
	
	# Plot 3: Green Wave Analysis
	fig2, ax3 = plt.subplots(figsize=(10, 6))
	# Plot traffic light states over distance and time
	for i, pos in enumerate(light_positions):
		# Draw horizontal line at traffic light position
		ax3.axhline(y=pos, color='grey', linestyle='--')
		ax3.text(0, pos + 50, f'TL{i + 1}', va='center')
		
		# Plot green and red periods
		cycle_time = green_durations[i] + red_durations[i]
		num_cycles = int(120 / cycle_time) + 1
		
		# Start with green (or other offset based on optimization)
		is_green = True
		start_time = 0
		
		for cycle in range(num_cycles):
			end_green = start_time + green_durations[i]
			end_red = end_green + red_durations[i]
			
			# Green period
			ax3.axvspan(start_time, end_green, ymin=pos / max_position - 0.05, ymax=pos / max_position + 0.05,
			            alpha=0.3, color='green')
			
			# Red period
			ax3.axvspan(end_green, end_red, ymin=pos / max_position - 0.05, ymax=pos / max_position + 0.05,
			            alpha=0.3, color='red')
			
			start_time = end_red
	if vehicle_trajectories:
		import matplotlib.cm as cm
		import matplotlib.colors as mcolors
		
		num_vehicles = len(vehicle_trajectories)
		vehicle_colors = cm.viridis(np.linspace(0,1,num_vehicles))
		
		for idx, (vehicle_id, trajectory) in enumerate(vehicle_trajectories.items()):
			if trajectory:
				times, positions = zip(*trajectory)
				ax3.plot(times, positions, label=f'Vehicle{vehicle_id}', color = vehicle_colors [idx], linewidth = 1.5)

	# Configure the plot
	ax3.set_xlim([0, 120])
	ax3.set_ylim([0, max_position])
	ax3.set_xlabel('Time (seconds)')
	ax3.set_ylabel('Position (meters)')
	ax3.set_title('Green Wave Analysis')
	
	#ax3.legend()
	ax3.grid(True)
	
	# Add summary information to the figure
	green_str = str([round(g, 1) for g in green_durations])
	red_str = str([round(r, 1) for r in red_durations])
	
	fig.suptitle(f"Traffic Simulation Results\nGreen: {green_str}, Red: {red_str}")
	
	# Adjust layout and show plots
	plt.tight_layout()
	
	# Save the figures
	fig.savefig('traffic_light_timing.png')
	fig2.savefig('green_wave_analysis.png')
	plt.show()
	''''''
	# fig2, ax3 = plt.subplots(figsize=(10, 6))
	#
	# # Plot ideal trajectory (vehicle moving at constant speed)
	# vehicle_speed = 14  # meters per second
	# times = np.linspace(0, 120, 1000)
	# positions = times * vehicle_speed
	# ax3.plot(times, positions, label='Ideal trajectory', color='blue')
	#
	# # Plot traffic light states over distance and time
	# for i, pos in enumerate(light_positions):
	# 	# Draw horizontal line at traffic light position
	# 	ax3.axhline(y=pos, color='grey', linestyle='--')
	# 	ax3.text(0, pos + 50, f'TL{i + 1}', va='center')
	#
	# 	# Plot green and red periods
	# 	cycle_time = green_durations[i] + red_durations[i]
	# 	num_cycles = int(120 / cycle_time) + 1
	#
	# 	# Start with green (or other offset based on optimization)
	# 	is_green = True
	# 	start_time = 0
	#
	# 	for cycle in range(num_cycles):
	# 		end_green = start_time + green_durations[i]
	# 		end_red = end_green + red_durations[i]
	#
	# 		# Green period
	# 		ax3.axvspan(start_time, end_green, ymin=pos / max_position - 0.05, ymax=pos / max_position + 0.05,
	# 		            alpha=0.3, color='green')
	#
	# 		# Red period
	# 		ax3.axvspan(end_green, end_red, ymin=pos / max_position - 0.05, ymax=pos / max_position + 0.05,
	# 		            alpha=0.3, color='red')
	#
	# 		start_time = end_red
	#
	# # Configure the plot
	# ax3.set_xlim([0, 120])
	# ax3.set_ylim([0, max_position])
	# ax3.set_xlabel('Time (seconds)')
	# ax3.set_ylabel('Position (meters)')
	# ax3.set_title('Green Wave Analysis')
	# ax3.legend()
	# ax3.grid(True)
	#
	# # Add summary information to the figure
	# green_str = str([round(g, 1) for g in green_durations])
	# red_str = str([round(r, 1) for r in red_durations])
	#
	# fig.suptitle(f"Traffic Simulation Results\nGreen: {green_str}, Red: {red_str}")
	#
	# # Adjust layout and show plots
	# plt.tight_layout()
	#
	# # Save the figures
	# fig.savefig('traffic_light_timing.png')
	# fig2.savefig('green_wave_analysis.png')
	# plt.show()


def run_visual_simulation(simulation_type, num_vehicles=3, mean_spawn_time=2.0, spawn_variance=0.5):
	# optimization_applied= False
	#
	optimization_applied = False
	optimized_green_durations = []
	optimized_red_durations = []
	
	vehicle_trajectories={}
	elapsed_time = 0
	max_time = 400
	
	"""Run a visual simulation with traffic light control and enhanced metrics"""
	metrics = Metrics()
	scenario_index = metrics.reset_density_flow_data()
	
	system = TrafficSystem()
	vehicles_to_spawn = num_vehicles  # Track how many vehicles we need to spawn
	
	road_length = 200
	# system.line_with_light(road_length, [30, 90, 200])
	
	# Initialize optimizer variables
	optimizer = None
	optimize_lights = False
	optimization_results = None
	# vehicle_velocity = 14  # Default velocity
	
	# Ask the user if they want to use normal distribution spawning for line cases
	use_normal_dist = False
	if simulation_type in ["line", "line_with_light"]:
		try:
			use_normal_dist_input = input(
				"\nUse normal distribution spawner for line simulation? (y/n, default=y): ").lower()
			# Default to yes for line simulations
			use_normal_dist = use_normal_dist_input != 'n'
			
			if use_normal_dist:
				print("Using normal distribution spawner")
				try:
					# If normal distribution is chosen, ask for number of vehicles to spawn
					num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
					vehicles_to_spawn = num_vehicles
					print(f"Using number of vehicles: {num_vehicles}")
				except ValueError:
					print(f"Invalid input. Using default number of vehicles: {num_vehicles}")
				
				# Ask for vehicle velocity
				try:
					vehicle_velocity = float(input("Enter vehicle velocity (default=1.4): ") or vehicle_velocity)
					print(f"Using vehicle velocity: {vehicle_velocity}")
				except ValueError:
					print(f"Invalid input. Using default velocity: {vehicle_velocity}")
				
				# Get spawn parameters
				try:
					print("\nSpawn Distribution Parameters:")
					mean_spawn_time = float(input("Enter mean spawn time (default=2.0): ") or mean_spawn_time)
					spawn_variance = float(input("Enter spawn variance (default=0.5): ") or spawn_variance)
					print(f"Using mean spawn time: {mean_spawn_time}, variance: {spawn_variance}")
				except ValueError:
					print(f"Invalid input. Using default values: mean={mean_spawn_time}, variance={spawn_variance}")
			else:
				print("Using regular spawning (when spawn location is empty)")
				try:
					# If regular spawning is chosen, ask for number of vehicles
					num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
					vehicles_to_spawn = num_vehicles
					print(f"Using number of vehicles: {num_vehicles}")
				except ValueError:
					print(f"Invalid input. Using default number of vehicles: {num_vehicles}")
				
				# Ask for vehicle velocity
				try:
					vehicle_velocity = float(input("Enter vehicle velocity (default=0.4): ") or vehicle_velocity)
					print(f"Using vehicle velocity: {vehicle_velocity}")
				except ValueError:
					print(f"Invalid input. Using default velocity: {vehicle_velocity}")
		
		except ValueError:
			print("Invalid input. Using normal distribution spawner.")
			use_normal_dist = True
	else:
		# For non-line simulations
		try:
			num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
			vehicles_to_spawn = num_vehicles
			print(f"Using number of vehicles: {num_vehicles}")
		except ValueError:
			print(f"Invalid input. Using default number of vehicles: {num_vehicles}")
	
	# Initialize system based on simulation type
	if simulation_type == "line":
		system.line(30)
		light = TrafficLight(0, "red", 10)
		system.add_traffic_light(light)
		# Add initial vehicle only if not using normal distribution
		if not use_normal_dist:
			# Add first vehicle to start the simulation
			vehicle = Vehicle(id=1, position=0, length=2, velocity=0)
			system.add_vehicle(vehicle)
	
	elif simulation_type == "line_with_light":
		road_length = 200
		light_positions = [30, 90, 150]
		system.line_with_light(road_length, light_positions)
		
		# Ask user if they want to use traffic optimization
		try:
			use_optimization = input("\nUse traffic light optimization? (y/n, default=n): ").lower()
			optimize_lights = use_optimization == 'y'
			
			if optimize_lights:
				print("Traffic light optimization will be applied.")
				
				# Create an optimizer instance
				optimizer = TrafficLightOptimizer(
					light_positions=light_positions,
					velocity=vehicle_velocity,
					min_green=10,
					max_green=30,
					min_red=5,
					max_red=15
				)
				
				print("Light positions:", light_positions)
				print("Vehicle velocity for optimization:", vehicle_velocity)
				print("Optimization will be run when simulation starts.")
			else:
				print("Traffic light optimization will not be used.")
		except ValueError:
			print("Invalid input. Traffic light optimization will not be used.")
		
		# Add initial vehicle only if not using normal distribution
		if not use_normal_dist:
			# Add first vehicle to start the simulation
			vehicle = Vehicle(id=1, position=0, length=2, velocity=0)
			system.add_vehicle(vehicle)
	
	elif simulation_type == "square":
		system.square(10)
		for i in range(num_vehicles):
			spacing = 1  # Distance between vehicles
			vehicle = Vehicle(id=i + 1, position=i * spacing, length=2, velocity=0)
			system.add_vehicle(vehicle)
	
	elif simulation_type == "grid":
		# Create a grid network
		rows, cols = 3, 3
		road_length = 10
		system.grid_network(rows, cols, road_length)
		
		# Use the method to add random vehicles - grid always uses fixed spawning
		system.add_random_vehicles(num_vehicles, velocity=0)
	
	controller = TrafficLightController(system)
	# Use optimization for grid - it works better for complex intersections
	# controller.set_control_mode("fixed")
	
	clock = pygame.time.Clock()
	running = True
	paused = False
	show_help = False
	simulation_started = False
	simulation_ended = False
	
	# Variables for the normal distribution spawning - only for line cases
	using_normal_distribution = (simulation_type in ["line", "line_with_light"]) and use_normal_dist
	# Variables for the regular spawning - only for line cases
	using_regular_spawning = (simulation_type in ["line", "line_with_light"]) and not use_normal_dist
	
	# Initialize variables for regular spawning
	if using_regular_spawning:
		system.regular_spawn = True
		system.regular_spawn_count = 1  # We already spawned one vehicle
		system.regular_spawn_target = num_vehicles
		system.regular_spawn_cooldown = 1.0  # Initial cooldown before checking for empty spawn
		system.regular_spawn_min_distance = 3.0  # Minimum distance from start to consider spawn area empty
		system.time_since_last_check = 0
	
	# Custom simulation function for optimization
	
	def simulation_func(green_durations, red_durations, delays, spawn_times, total_vehicles):
		"""
		Run a simulation with the given parameters and return metrics.
		This function is used by the optimizer to evaluate different light timings.
		"""
		# Create a temporary system for simulation
		temp_system = TrafficSystem()
		temp_system.line_with_light(road_length, light_positions)
		
		# Configure traffic lights with the given parameters
		for i, light in enumerate(temp_system.lights):
			if i < len(green_durations) and i < len(red_durations) and i < len(delays):
				light.cycle_duration = green_durations[i] + red_durations[i]
				light.timer = delays[i] % light.cycle_duration
				light.state = "green"  # Start with green
		
		# Create vehicles based on spawn times
		for i, spawn_time in enumerate(spawn_times):
			vehicle = Vehicle(id=i + 1, position=0, length=1, velocity=vehicle_velocity)
			temp_system.add_vehicle(vehicle)
		
		# Run the simulation for a fixed time
		sim_time = 0
		dt = 0.1
		max_sim_time = 200  # Maximum simulation time
		
		# Keep track of exited vehicles and stops
		exited_vehicles = 0
		stops_per_light = [0] * len(temp_system.lights)
		
		# Track which vehicles have been stopped at each light
		vehicles_stopped_at_light = {i: set() for i in range(len(temp_system.lights))}
		
		while sim_time < max_sim_time and (exited_vehicles < total_vehicles or temp_system.vehicles):
			# Update all vehicles
			vehicles_to_remove = []
			for vehicle in temp_system.vehicles:
				# Check for stops at traffic lights
				for i, light in enumerate(temp_system.lights):
					if (vehicle.waiting or vehicle.stopped) and vehicle._is_at_light(light, temp_system):
						if vehicle.id not in vehicles_stopped_at_light[i]:
							vehicles_stopped_at_light[i].add(vehicle.id)
							stops_per_light[i] += 1
				
				# Update vehicle
				if vehicle.update(dt, temp_system):
					vehicles_to_remove.append(vehicle)
			
			# Remove vehicles that have reached the end
			for vehicle in vehicles_to_remove:
				if vehicle in temp_system.vehicles:
					temp_system.vehicles.remove(vehicle)
					exited_vehicles += 1
			
			# Update traffic lights
			for i, light in enumerate(temp_system.lights):
				if i < len(green_durations) and i < len(red_durations):
					# Custom update logic based on the provided durations
					light.timer += dt
					if light.state == "green" and light.timer >= green_durations[i]:
						light.state = "red"
						light.timer = 0
					elif light.state == "red" and light.timer >= red_durations[i]:
						light.state = "green"
						light.timer = 0
			
			# Advance simulation time
			sim_time += dt
			
			# If all vehicles have exited, we can stop
			if exited_vehicles >= total_vehicles and not temp_system.vehicles:
				break
		
		# Compile metrics
		results = {
			'exited_vehicles': exited_vehicles,
			'stops_per_light': stops_per_light,
			'total_stops': sum(stops_per_light),
			'simulation_time': sim_time
		}
		
		return results
	
	def start_simulation():
		metrics.set_checkpoint_position(system)
		nonlocal simulation_started, optimization_applied, optimized_green_durations, optimized_red_durations
		
		if not simulation_started:
			
			# For traffic light optimization in line_with_light
			if simulation_type == "line_with_light" and optimize_lights and optimizer:
				# Generate spawn times based on normal distribution
				print("\nGenerating spawn times based on parameters...")
				spawn_times = []
				
				# Generate spawn times based on mean_spawn_time and variance
				current_time = 0
				for i in range(num_vehicles):
					# Use normal distribution for time intervals
					interval = max(0.1, np.random.normal(mean_spawn_time, np.sqrt(spawn_variance)))
					current_time += interval
					spawn_times.append(current_time)
				
				print(f"Generated {len(spawn_times)} spawn times.")
				
				# Run optimization
				print("\nRunning traffic light optimization...")
				optimization_start = time.time()
				
				optimization_results = optimizer.optimize(
					spawn_times=spawn_times,
					total_vehicles=num_vehicles,
					simulation_function=simulation_func,
					max_iter=15,
					pop_size=10
				)
				
				optimization_time = time.time() - optimization_start
				print(f"Optimization completed in {optimization_time:.2f} seconds")
				
				# Apply optimized timings to traffic lights
				green_durations = optimization_results['green_durations']
				red_durations = optimization_results['red_durations']
				delays = optimization_results['delays']
				
				# Store the optimized durations for later use
				nonlocal optimized_green_durations, optimized_red_durations
				optimized_green_durations = green_durations.copy()  # Make a copy to be safe
				optimized_red_durations = red_durations.copy()
				
				# print("\nApplying optimized light timings:")
				# for i, light in enumerate(system.lights):
				# 	if i < len(green_durations):
				# 		light.cycle_duration = green_durations[i] + red_durations[i]
				# 		light.timer = delays[i] % light.cycle_duration
				# 		light.state = "green"  # Start with green
				# 		print(
				# 			f"Light {i}: Green={green_durations[i]:.2f}s, Red={red_durations[i]:.2f}s, Delay={delays[i]:.2f}s")
				#
				# Mark that optimization has been applied
				# nonlocal optimization_applied
				optimization_applied = True
				controller.current_mode = "DE Optimized"
				print("DE optimization applied - controller will be bypassed")
				# Initialize all lights with clean states
				for i, light in enumerate(system.lights):
					if i < len(green_durations) and i < len(red_durations):
						light.cycle_duration = green_durations[i] + red_durations[i]
						light.timer = 0  # Reset timer
						
						# Stagger the light states - alternative green and red
						light.state = "green" if i % 2 == 0 else "red"
						# Stagger the light states - don't start all green
						# if i % 2 == 0:
						# 	light.state = "green"
						# else:
						# 	light.state = "red"
						
						print(
							f"Light {i}: Initial state={light.state}, Green={green_durations[i]:.2f}s, Red={red_durations[i]:.2f}s")
			
			# Set vehicle velocities based on the simulation type
			metrics.set_checkpoint_position(system)
			for vehicle in system.vehicles:
				if simulation_type in ["line", "line_with_light"]:
					vehicle.velocity = vehicle_velocity
				elif simulation_type == "square":
					vehicle.velocity = 0.4
				elif simulation_type == "grid":
					vehicle.velocity = 0.3
			
			# For line simulations with normal distribution selected
			if using_normal_distribution:
				# Initialize the continuous normal distribution spawning
				system.continuous_spawn = True
				system.total_spawned = 0  # Add counter for tracking spawned vehicles
				system.target_spawns = num_vehicles  # Set target number of spawns
				system.spawn_vehicles_continuously_with_normal_distribution(
					mean_time_between_spawns=mean_spawn_time,
					variance=spawn_variance,
					velocity=vehicle_velocity,
					length=1,
					minimum_distance=1.0
				)
			# else:
			# 	# For all other cases, set velocities as before
			# 	for vehicle in system.vehicles:
			# 		if simulation_type == "line" or simulation_type == "line_with_light":
			# 			vehicle.velocity = 0.4
			# 		elif simulation_type == "square":
			# 			vehicle.velocity = 0.4
			# 		elif simulation_type == "grid":
			# 			vehicle.velocity = 0.3
			simulation_started = True
	
	def reset_simulation():
		nonlocal system, controller, simulation_started, simulation_ended, paused, metrics, scenario_index
		nonlocal vehicles_to_spawn, using_normal_distribution, using_regular_spawning, use_normal_dist
		nonlocal num_vehicles, mean_spawn_time, spawn_variance, vehicle_velocity, optimizer, optimize_lights
		nonlocal optimization_applied, optimized_green_durations, optimized_red_durations
		
		# Reset optimization flags
		optimization_applied = False
		optimized_green_durations = []
		optimized_red_durations = []
		
		# Ask the user if they want to use normal distribution spawning for line cases
		if simulation_type in ["line", "line_with_light"]:
			try:
				use_normal_dist_input = input(
					"\nUse normal distribution spawner for line simulation? (y/n, default=y): ").lower()
				# Default to yes for line simulations
				use_normal_dist = use_normal_dist_input != 'n'
				using_normal_distribution = use_normal_dist
				using_regular_spawning = not use_normal_dist
				
				if use_normal_dist:
					print("Using normal distribution spawner")
					try:
						# If normal distribution is chosen, ask for number of vehicles to spawn
						num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
						vehicles_to_spawn = num_vehicles
						print(f"Using number of vehicles: {num_vehicles}")
					except ValueError:
						print(f"Invalid input. Using default number of vehicles: {num_vehicles}")
					
					# Get spawn parameters
					try:
						mean_spawn_time = float(input("Enter mean spawn time (default=2.0): ") or mean_spawn_time)
						spawn_variance = float(input("Enter spawn variance (default=0.5): ") or spawn_variance)
						print(f"Using mean spawn time: {mean_spawn_time}, variance: {spawn_variance}")
					except ValueError:
						print(f"Invalid input. Using previous values.")
				else:
					print("Using regular spawning (when spawn location is empty)")
					try:
						# If regular spawning is chosen, ask for number of vehicles
						num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
						vehicles_to_spawn = num_vehicles
						print(f"Using number of vehicles: {num_vehicles}")
					except ValueError:
						print(f"Invalid input. Using default number of vehicles: {num_vehicles}")
			except ValueError:
				print("Invalid input. Using normal distribution spawner.")
				use_normal_dist = True
				using_normal_distribution = True
				using_regular_spawning = False
		else:
			# For non-line simulations
			try:
				num_vehicles = int(input("Enter number of vehicles to spawn (default=3): ") or num_vehicles)
				vehicles_to_spawn = num_vehicles
				print(f"Using number of vehicles: {num_vehicles}")
			except ValueError:
				print(f"Invalid input. Using previous number of vehicles: {num_vehicles}")
		
		# Reset flags
		simulation_started = False
		simulation_ended = False
		paused = False
		vehicles_to_spawn = num_vehicles
		
		# Reset metrics
		metrics = Metrics()
		scenario_index = metrics.reset_density_flow_data()
		
		# Reinitialize the system
		system = TrafficSystem()
		
		# Recreate the simulation setup based on simulation_type
		if simulation_type == "line":
			system.line(30)
			light = TrafficLight(0, "red", 10)
			system.add_traffic_light(light)
			# Add initial vehicle only if not using normal distribution
			if not use_normal_dist:
				# Add first vehicle to start the simulation
				vehicle = Vehicle(id=1, position=0, length=2, velocity=0)
				system.add_vehicle(vehicle)
		
		elif simulation_type == "line_with_light":
			system.line_with_light(road_length, light_positions)
			# Add initial vehicle only if not using normal distribution
			if not use_normal_dist:
				# Add first vehicle to start the simulation
				vehicle = Vehicle(id=1, position=0, length=2, velocity=0)
				system.add_vehicle(vehicle)
				
				# Ask user if they want to use traffic optimization again
				try:
					use_optimization = input("\nUse traffic light optimization? (y/n, default=n): ").lower()
					optimize_lights = use_optimization == 'y'
					
					if optimize_lights:
						print("Traffic light optimization will be applied.")
						
						# Recreate optimizer with updated parameters
						optimizer = TrafficLightOptimizer(
							light_positions=light_positions,
							velocity=vehicle_velocity,
							min_green=5,
							max_green=15,
							min_red=2,
							max_red=10
						)
						
						print("Light positions:", light_positions)
						print("Vehicle velocity for optimization:", vehicle_velocity)
						print("Optimization will be run when simulation starts.")
					else:
						print("Traffic light optimization will not be used.")
				except ValueError:
					print("Invalid input. Traffic light optimization will not be used.")
		
		
		
		
		elif simulation_type == "square":
			system.square(10)
			for i in range(num_vehicles):
				spacing = 3  # Distance between vehicles
				vehicle = Vehicle(id=i + 1, position=i * spacing, length=2, velocity=0)
				system.add_vehicle(vehicle)
		
		elif simulation_type == "grid":
			# Create a grid network
			rows, cols = 3, 3
			road_length = 10
			system.grid_network(rows, cols, road_length)
			
			# Add vehicles with destinations - grid always uses fixed spawning
			system.add_random_vehicles(num_vehicles, velocity=0)
		
		# Initialize variables for regular spawning
		if using_regular_spawning:
			system.regular_spawn = True
			system.regular_spawn_count = 1  # We already spawned one vehicle
			system.regular_spawn_target = num_vehicles
			system.regular_spawn_cooldown = 1.0  # Initial cooldown before checking for empty spawn
			system.regular_spawn_min_distance = 3.0  # Minimum distance from start to consider spawn area empty
			system.time_since_last_check = 0
		
		# Reset the controller
		controller = TrafficLightController(system)
		
		# Debug: Check traffic lights before optimization
		print(f"LIGHT COUNT BEFORE OPTIMIZATION: {len(system.lights)}")
		for i, light in enumerate(system.lights):
			print(f"Light {i}: position={getattr(light, 'position', (light.x, light.y))}")
		
		# If optimization is enabled, run it before starting the simulation
		if optimize_lights and optimizer:
			print("Running optimization...")
			
			# Generate simple spawn times (e.g., evenly spaced)
			spawn_times = [i * mean_spawn_time for i in range(vehicles_to_spawn)]
			
			optimization_results = optimizer.optimize(
				spawn_times=spawn_times,
				total_vehicles=vehicles_to_spawn,
				simulation_function=simulation_func
			)
			
			optimized_green_durations = optimization_results["best_green_durations"]
			optimized_red_durations = optimization_results["best_red_durations"]
			
			print(f"Optimized Green Durations: {optimized_green_durations}")
			print(f"Optimized Red Durations: {optimized_red_durations}")
			
			# Apply DE durations to simulation's actual traffic lights
			for i, light in enumerate(system.lights):
				light.green_duration = optimized_green_durations[i]
				light.red_duration = optimized_red_durations[i]
				light.cycle_duration = light.green_duration + light.red_duration
				light.timer = 0
				light.state = "green"
			
			optimization_applied = True
		
		# controller.set_control_mode("fixed")
		
		# if simulation_type == "line_with_light" and optimize_lights:
		# 	# Recreate optimizer with updated parameters
		# 	optimizer = TrafficLightOptimizer(
		# 		light_positions=light_positions,
		# 		velocity=vehicle_velocity,
		# 		min_green=5,
		# 		max_green=15,
		# 		min_red=2,
		# 		max_red=10
		# 	)
		
		# Add this new function to draw optimization results
		
		def draw_optimization_info(screen, system, optimization_results):
			
			if not optimization_results or simulation_type != "line_with_light" or not optimize_lights:
				return
			
			info_font = pygame.font.SysFont(None, 18)
			panel_x = SCREEN_WIDTH - 220
			panel_y = 200
			
			# Panel title
			title_text = font.render("Optimization Results:", True, (0, 0, 0))
			screen.blit(title_text, (panel_x, panel_y))
			panel_y += 30
			
			# Show green and red durations for each light
			if len(optimized_green_durations) > 0 and len(optimized_red_durations) > 0:
				for i in range(len(optimized_green_durations)):
					if i < len(system.lights):
						light_info = f"Light {i}: G={optimized_green_durations[i]:.1f}s, R={optimized_red_durations[i]:.1f}s"
						info_text = info_font.render(light_info, True, (0, 0, 0))
						screen.blit(info_text, (panel_x, panel_y))
						panel_y += 20
				
				# Add information about stops
				stops_per_light = system.detect_stops_at_traffic_lights()
				total_stops = sum(stops_per_light)
				stop_info = f"Total stops: {total_stops}"
				info_text = info_font.render(stop_info, True, (0, 0, 0))
				screen.blit(info_text, (panel_x, panel_y))
			'''
			#
			# # Show green and red durations for each light
			# green_durations = optimization_results['green_durations']
			# red_durations = optimization_results['red_durations']
			#
			# for i in range(len(green_durations)):
			# 	if i < len(system.lights):
			# 		light_info = f"Light {i}: G={green_durations[i]:.1f}s, R={red_durations[i]:.1f}s"
			# 		info_text = info_font.render(light_info, True, (0, 0, 0))
			# 		screen.blit(info_text, (panel_x, panel_y))
			# 		panel_y += 20
			#
			# # Show objective value
			# if 'objective_value' in optimization_results:
			# 	obj_text = info_font.render(f"Objective: {optimization_results['objective_value']:.2f}", True,
			# 	                            (0, 0, 0))
			# 	screen.blit(obj_text, (panel_x, panel_y))
			#
		'''
	
	start_button = Button(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 50, 100, 40, "START", start_simulation)
	
	# Prepare to track stops during the main simulation
	stops_at_lights = [0] * len(system.lights)
	vehicles_stopped_at_light = {i: set() for i in range(len(system.lights))}
	
	while running:
		mouse_pos = pygame.mouse.get_pos()
		start_button.check_hover(mouse_pos)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					paused = not paused
				elif event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_h:
					show_help = not show_help
				elif event.key == pygame.K_1 and not optimization_applied:
					controller.set_control_mode("fixed")
				elif event.key == pygame.K_2 and not optimization_applied:
					controller.set_control_mode("green_wave")
				elif event.key == pygame.K_3 and not optimization_applied:
					controller.set_control_mode("optimization")
				elif event.key == pygame.K_r:
					reset_simulation()
			
			# Handle start button
			if not simulation_started and not simulation_ended:
				start_button.handle_event(event)
		
		screen.fill(BG_COLOR)
		
		if not paused and simulation_started and not simulation_ended:
			elapsed_time += 0.1
			# 	if elapsed_time % 1.0 < 0.1:  # Print once per second approximately
			# 		print(f"Time {elapsed_time:.1f} - Light states:")
			# 		for i, light in enumerate(system.lights):
			# 			#print(f"  Light {i}: State={light.state}, Timer={light.timer:.1f}/{optimized_green_durations[i] if light.state == 'green' else optimized_red_durations[i]:.1f}")
			
			# For line simulations with normal distribution, handle spawning differently
			if using_normal_distribution:
				# Check if we should stop spawning new vehicles (reached target count)
				if hasattr(system, 'total_spawned') and hasattr(system, 'target_spawns'):
					if system.total_spawned >= system.target_spawns:
						system.continuous_spawn = False  # Stop spawning
				
				# Continuous spawn system updates
				if hasattr(system, 'continuous_spawn') and system.continuous_spawn:
					check_for_spawn(system, 0.1)
					system.update(0.1)  # Normal distribution spawning happens inside the update
					system.update(0.1, maintain_count=num_vehicles)
				else:
					system.update(0.1)  # For line with fixed spawning
			
			# controller.update(0.1)
			if optimization_applied:
				# Custom direct update for lights that bypasses the controller completely
				for i, light in enumerate(system.lights):
					if i < len(optimized_green_durations) and i < len(optimized_red_durations):
						# Store previous state to detect changes
						prev_state = light.state
						light.timer += 0.1
						
						if light.state == "green" and light.timer >= optimized_green_durations[i]:
							light.state = "red"
							light.timer = 0
							if elapsed_time % 5 < 0.1:  # Print only occasionally
								print(f"Light {i} changing from green to red at time {elapsed_time:.1f}")
						elif light.state == "red" and light.timer >= optimized_red_durations[i]:
							light.state = "green"
							light.timer = 0
							if elapsed_time % 5 < 0.1:  # Print only occasionally
								print(f"Light {i} changing from red to green at time {elapsed_time:.1f}")
						
						for i, light in enumerate(system.lights):
							for vehicle in system.vehicles:
								# Check if the vehicle is at the light AND the light is red (crucial condition)
								if vehicle._is_at_light(light, system) and light.state == "red":
									# If vehicle is not already counted as stopped at this light
									if vehicle.id not in vehicles_stopped_at_light[i]:
										# Only count it if it's actually stopped or waiting
										if vehicle.waiting or vehicle.stopped:
											vehicles_stopped_at_light[i].add(vehicle.id)
											stops_at_lights[i] += 1
											if elapsed_time % 10 < 0.1:  # Log occasionally
												print(
													f"Vehicle {vehicle.id} stopped at light {i}, total stops: {stops_at_lights[i]}")
			else:
				# If not optimized, use the controller
				controller.update(0.1)
				
			# Track vehicle positions for trajectories
			for vehicle in system.vehicles:
				if vehicle.id not in vehicle_trajectories:
					vehicle_trajectories[vehicle.id] = []
				
				# Add current time and position to the trajectory
				vehicle_trajectories[vehicle.id].append((system.time, vehicle.position))
			
				# Track stops at lights - with the correct condition checking
				for i, light in enumerate(system.lights):
					for vehicle in system.vehicles:
						# Check if the vehicle is at the light AND the light is red
						if vehicle._is_at_light(light, system) and light.state == "red":
							# If vehicle is not already counted as stopped at this light
							if vehicle.id not in vehicles_stopped_at_light[i]:
								# Only count it if it's actually stopped or waiting
								if vehicle.waiting or vehicle.stopped:
									vehicles_stopped_at_light[i].add(vehicle.id)
									stops_at_lights[i] += 1
									if elapsed_time % 10 < 0.1:  # Log occasionally
										print(
											f"Vehicle {vehicle.id} stopped at light {i}, total stops: {stops_at_lights[i]}")
										
			# Rest of the existing update code...
			
			if elapsed_time >= max_time:
				simulation_ended = True
				print("Simulation reached maximum time limit")
				
				# When simulation ends, calculate and display final metrics
				total_vehicles = getattr(system, 'total_spawned', num_vehicles)
				exited_vehicles = getattr(system, 'vehicles_exited', 0)
				# total_stops = sum(system.detect_stops_at_traffic_lights())
				total_stops = sum(stops_at_lights)
				
				print(f"\nFinal Metrics:")
				print(f"Total vehicles: {total_vehicles}")
				print(f"Exited vehicles: {exited_vehicles}")
				print(f"Completion rate: {exited_vehicles / total_vehicles * 100:.1f}%")
				print(f"Total stops: {total_stops}")
				print(f"Average stops per vehicle: {total_stops / total_vehicles:.2f}")
				
				# End of simulation
				# if optimization_applied:
				# 	light_positions = [light.position for light in system.lights]
				# 	plot_optimization_results(
				# 		optimized_green_durations,
				# 		optimized_red_durations,
				# 		light_positions,
				# 		stops_at_lights
				# 	)
				#
			metrics.collect_density_data(system)
			metrics.collect_flow_data(system, point_position=10, time_interval=1, scenario_index=scenario_index)
			
			# For line simulations with normal distribution, check if simulation should end
			if using_normal_distribution:
				# End if we've spawned the target number of vehicles and they've all reached their destination
				if (hasattr(system, 'total_spawned') and system.total_spawned >= system.target_spawns
						and len(system.vehicles) == 0):
					simulation_ended = True
					print(f"Simulation completed! All {system.total_spawned} vehicles have reached their destination.")
			# For other simulations, use the existing check
			elif check_simulation_ended(system, simulation_type):
				simulation_ended = True
				print("Simulation completed! All vehicles have reached their destination.")
		
		# Draw the appropriate simulation visualization
		if simulation_type == "line" or simulation_type == "line_with_light":
			draw_line_simulation(screen, system)
		elif simulation_type == "square":
			draw_square_simulation(screen, system)
		elif simulation_type == "grid":
			draw_grid_simulation(screen, system)
		
		# Show what control mode is active
		# Show what control mode is active
		control_mode_text = "DE Optimized" if optimization_applied else controller.current_mode
		control_info = font.render(f"Control Mode: {control_mode_text}", True, (0, 0, 0))
		screen.blit(control_info, (10, 70))
		
		draw_simulation_info(screen, system, None, simulation_started, simulation_ended)
		
		# Draw optimization info if available
		if optimization_results and simulation_type == "line_with_light" and optimize_lights:
			draw_optimization_info(screen, system, optimization_results)
		
		# Draw controls and UI elements
		if paused and simulation_started:
			pause_text = font.render("PAUSED (SPACE to resume)", True, (200, 0, 0))
			screen.blit(pause_text, (SCREEN_WIDTH // 2 - 150, 10))
		elif simulation_started and not simulation_ended:
			pause_text = font.render("SPACE to pause", True, (0, 0, 0))
			screen.blit(pause_text, (SCREEN_WIDTH // 2 - 80, 10))
		
		if not simulation_started and not simulation_ended:
			start_button.draw(screen)
		elif simulation_ended:
			completed_text = font.render("Simulation Complete", True, (0, 150, 0))
			completed_rect = completed_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
			screen.blit(completed_text, completed_rect)
		
		if show_help:
			draw_help_menu(screen)
		else:
			hint_text = font.render("Press H for controls", True, (0, 0, 0))
			screen.blit(hint_text, (SCREEN_WIDTH - 200, 10))
		
		pygame.display.flip()
		dt = 0.1
		clock.tick(60)
	
	metrics.save_density_flow_data()
	metrics.plot_density_data(system)
	metrics.plot_velocity_data()
	
	light_positions = [light.position for light in system.lights]
	print(f"Accumulated stops at lights: {stops_at_lights}")
	
	if optimization_applied:
		# Use the optimized durations
		print(
			f"Plotting with optimized durations: {len(optimized_green_durations)} green durations and {len(light_positions)} lights")
		plot_optimization_results(
			green_durations=optimized_green_durations,
			red_durations=optimized_red_durations,
			light_positions=light_positions,
			stops_at_lights=stops_at_lights,
			vehicle_trajectories = vehicle_trajectories  # Add this parameter
		)
		
	else:
		# When not using optimization, use the default cycle durations from the controller or lights
		green_durations = []
		red_durations = []
		
		for light in system.lights:
			# Default split: 50% green, 50% red from the light's cycle_duration
			if hasattr(light, 'cycle_duration'):
				green_duration = light.cycle_duration / 2
				red_duration = light.cycle_duration / 2
			else:
				# Default values if cycle_duration isn't set
				green_duration = 15.0
				red_duration = 15.0
			
			green_durations.append(green_duration)
			red_durations.append(red_duration)
		
		print(
			f"Plotting with default durations: {len(green_durations)} green durations and {len(light_positions)} lights")
		plot_optimization_results(
			green_durations=green_durations,
			red_durations=red_durations,
			light_positions=light_positions,
			stops_at_lights=stops_at_lights,
			vehicle_trajectories=vehicle_trajectories  # Add this parameter
		)
		
	
	pygame.quit()


# Add this new function to handle the continuous spawning
def check_for_spawn(system, dt):
	"""
	Check if it's time to spawn a new vehicle in continuous spawn mode.

	Parameters:
	-----------
	system : TrafficSystem
		The traffic system instance
	dt : float
		Time increment
	"""
	# Update the time since last spawn
	system.time_since_last_spawn += dt
	
	# Check if enough time has passed to spawn another vehicle
	if system.time_since_last_spawn >= system.next_spawn_time:
		# Check if spawn area is clear
		spawn_area_clear = True
		for vehicle in system.vehicles:
			if vehicle.position < system.spawn_length + system.spawn_min_distance:
				spawn_area_clear = False
				break
		
		if spawn_area_clear:
			# Get the next ID
			next_id = max([v.id for v in system.vehicles], default=0) + 1
			
			# Create a new vehicle at the start of the line
			vehicle = Vehicle(next_id,
			                  velocity=system.spawn_velocity,
			                  length=system.spawn_length,
			                  minimum_distance=system.spawn_min_distance)
			
			# For a line, the route is simply from start (0) to end (edge length)
			edge_length = system.edges[0][2]  # x2 coordinate of the first edge
			vehicle.position = 0
			vehicle.destination = edge_length
			vehicle.edge_id = 0  # The line has only one edge with ID 0
			
			system.add_vehicle(vehicle)
			
			# Increment the counter for successful spawns
			if not hasattr(system, 'total_spawned'):
				system.total_spawned = 0
			system.total_spawned += 1
			
			# Reset the time since last spawn
			system.time_since_last_spawn = 0
			
			# Generate the next spawn time from the normal distribution
			# Use max to ensure we don't get negative times
			std_dev = np.sqrt(system.spawn_variance)
			system.next_spawn_time = max(0.1, np.random.normal(system.spawn_mean, std_dev))
			
			print(f"Spawned vehicle {vehicle.id}")
			print(f"Next spawn in {system.next_spawn_time:.2f} time units")
			
			# If num_vehicles is defined in the global scope, use it, otherwise don't include this line
			if hasattr(system, 'num_vehicles'):
				print(f"Total spawned so far: {system.total_spawned}/{system.num_vehicles}")
			else:
				print(f"Total spawned so far: {system.total_spawned}")
			
			# If we've spawned enough vehicles, return a signal
			if hasattr(system, 'num_vehicles') and system.total_spawned >= system.num_vehicles:
				return True
		
		# Return False to indicate we should continue spawning
		return False


if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] in ["line", "line_with_light", "square", "grid"]:
		num_vehicles = 3  # Default
		if len(sys.argv) > 2:
			try:
				num_vehicles = int(sys.argv[2])
			except ValueError:
				print("Invalid number of vehicles, using default (3)")
		run_visual_simulation(sys.argv[1], num_vehicles)
	else:
		print("Select simulation type:")
		print("1: Line - Simple straight road")
		print("2: line_with_light - Line with traffic light")
		print("3: Square - Square circuit with traffic lights")
		print("4: Grid - Grid network with intersections")
		
		choice = input("Enter choice (1-4): ")
		
		sim_type = ""
		if choice == "1":
			sim_type = "line"
		elif choice == "2":
			sim_type = "line_with_light"
		elif choice == "3":
			sim_type = "square"
		elif choice == "4":
			sim_type = "grid"
		else:
			print("Invalid choice")
			sys.exit(1)
		
		try:
			num_vehicles = 1
			if num_vehicles < 1:
				print("Using default value (3) - input must be between 1-10")
				num_vehicles = 3
		except ValueError:
			print("Using default value (3) - please enter a number next time")
			num_vehicles = 3
		
		run_visual_simulation(sim_type, num_vehicles)