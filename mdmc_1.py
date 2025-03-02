import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Set NumPy to display numbers in standard format, not scientific notation
np.set_printoptions(suppress=True)

#set the constraint for discretization of first question
def initial_den(M,Nt,initial=20):
	P = np.zeros((M, Nt+1), dtype=int)
	# Cumulative cars passing each point
	P[0,0]=initial
	return P

def flux(vmax,pmax,p):
	vel=vmax*(1-(p/pmax))
	q=vel*p
	return q

def initial_flux(M,Nt,vmax,pmax,initial=20):
	Q=np.zeros((M,Nt+1), dtype=int)
	Q[0,0]=flux(vmax,pmax,initial)
	return Q



#set up euler method using forward differences method
def vectorisation_Q(M,Nt,dt,dx,initial,vmax,pmax):
	#Q=initial_flux(M,Nt,vmax,pmax)
	P=initial_den(M,Nt,initial)

	# Initialize flux array
	Q = np.zeros((M, Nt + 1))

	# Calculate initial flux
	for m in range(M):
		Q[m, 0] = flux(vmax, pmax, P[m, 0])

	# Simulation loop
	for t in range(Nt):
		# Update density for each position
		for m in range(M):
			# Calculate updated flux at current position
			Q[m, t] = flux(vmax, pmax, P[m, t])

			# First position (boundary condition: cars only leave, none enter)
			if m == 0 and t < Nt:
				P[m, t + 1] = P[m, t] - (dt / dx) * Q[m, t]

			# Interior positions
			elif m > 0 and t < Nt:
				P[m, t + 1] = P[m, t] - (dt / dx) * (Q[m, t] - Q[m - 1, t])

		for m in range(M):
			if t < Nt:
				Q[m, t + 1] = flux(vmax, pmax, P[m, t + 1])

	return Q, P

	#
	# for t in range(0, Nt):
	# 	for m in range(0, M):
	# 		Q[m, t] = flux(vmax, pmax, P[m, t])  # Calculate flux for all m, t
	#
	# 		if m == 1:
	# 			# Boundary condition at m = 1
	# 			P[m, t + 1] = (dt / dx) * (0 - Q[m, t]) + P[m, t]
	# 		elif m>0:
	# 			# Update density using the flux from previous and current positions
	# 			P[m, t + 1] = (dt / dx) * (Q[m - 1, t] - Q[m, t]) + P[m, t]
	#
	#
	# return Q,P


#question 1.a)
M=3
dx=1500
tmax=10*60
dt=90
Nt=int(tmax/dt)
vmax=13
pmax=100
initial=20

#question1a.i
Q,P=vectorisation_Q(M,Nt,dt,dx,initial,vmax,pmax)

# Print results
print("Flux at each position over time:")
print(Q)

print("Density at each position over time:")
print(P)

# Check if the simulation is stable (CFL condition)
courant = vmax * dt / dx
print(f"\nCFL condition: {courant} (should be < 1 for stability)")

# Plot results
time_points = np.linspace(0, tmax, Nt+1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for m in range(M):
    plt.plot(time_points, P[m, :], label=f"Position {m}")
plt.xlabel("Time (seconds)")
plt.ylabel("Density (cars/km)")
plt.title("Traffic Density")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for m in range(M):
    plt.plot(time_points, Q[m, :], label=f"Position {m}")
plt.xlabel("Time (seconds)")
plt.ylabel("Flux (cars/second)")
plt.title("Traffic Flux")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# # Calculate travel time between positions
# travel_time = dx/v
# print(f"\nExpected travel time between positions: {travel_time} seconds")
# print(f"Time step: {dt} seconds")
#
# # Plot the results
# plt.figure(figsize=(10, 6))
# time_points = np.linspace(0, tmax, Nt+1)
# for m in range(M):
#     plt.plot(time_points, Q[m, :], label=f"Position {m}")
#
# plt.xlabel("Time (seconds)")
# plt.ylabel("Number of cars")
# plt.title("Cars passing through traffic lights over time")
# plt.legend()
# plt.grid(True)
# plt.show()

