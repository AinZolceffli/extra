import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#set the constraint for discretization of first question
def initial_mat(M,Nt):
	Q = np.zeros((M, Nt+1))
	Q[0,0]=100

	return Q

#set up euler method using forward differences method
def vectorisation_Q(M,Nt,v,dt,dx):
	Q=initial_mat(M,Nt)
	for t in range(0, Nt):
		# Update first position (let cars move away from the origin)
		if t > 0:
			Q[0, t + 1] = Q[0, t] - (dt / dx) * v * (Q[0, t])

		# Update subsequent positions
		for m in range(1, M):
			if t < Nt:
				Q[m, t + 1] = Q[m, t] - (dt / dx) * v * (Q[m, t] - Q[m - 1, t])

	return Q


#question 1.a)
M=3
dx=1500
tmax=10*60
dt=90
Nt=int(tmax/dt)
v=15

#question1a.i
Q=vectorisation_Q(M,Nt,v,dt,dx)

# Print results
print("Traffic at each position over time:")
print(Q)

# Calculate travel time between positions
travel_time = dx/v
print(f"\nExpected travel time between positions: {travel_time} seconds")
print(f"Time step: {dt} seconds")

# Plot the results
plt.figure(figsize=(10, 6))
time_points = np.linspace(0, tmax, Nt+1)
for m in range(M):
    plt.plot(time_points, Q[m, :], label=f"Position {m}")

plt.xlabel("Time (seconds)")
plt.ylabel("Number of cars")
plt.title("Cars passing through traffic lights over time")
plt.legend()
plt.grid(True)
plt.show()

