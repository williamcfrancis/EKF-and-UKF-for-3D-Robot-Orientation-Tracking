import numpy as np
import matplotlib.pyplot as plt

x0 = np.random.normal(1, 2)

a = -1
D = []
for i in range(100):
	x_noise = np.random.normal(0, 1)
	y_noise = np.random.normal(0, 0.5)
	if(i==0):
		x = a*x0 + x_noise
		y = np.sqrt(x0**2 + 1) + y_noise
	else:
		x = a*x + x_noise
		y = np.sqrt(x**2 + 1) + y_noise
	D.append(y)

a0 = -10
R = np.array([[1, 0], 
			  [0, 0.01]])
Q = 0.5
mu_k = np.array([[x0], 
				[a0]])
sigma_k = np.array([[2.0, 0], 
					[0, 1.5]])
a, var_l, var_h = [], [], []
a.append(a0)
var_l.append(a0 - 1.5)
var_h.append(a0 + 1.5)

for i in range(100):
	y_k1 = D[i]
	# Propagating dynamics
	A = np.array([[mu_k[1, 0], mu_k[0, 0]], 
				  [0, 1]])
	mu_k1 = np.array([[mu_k[0, 0]*mu_k[1, 0]], 
					  [mu_k[1, 0]]])
	sigma_k1 = A @ sigma_k @ A.T + R # 2x2

	# Incorporatnig the observation
	C = np.array([[mu_k1[0, 0] / np.sqrt(mu_k1[0, 0]**2 + 1), 0]])
	K = (sigma_k1 @ C.T) / (C @ sigma_k1 @ C.T + Q) # 2x1
	y_k1_prime = y_k1 - np.sqrt(mu_k1[0, 0]**2 + 1) + (C @ mu_k1)

	# Update
	mu_k = mu_k1 + (K @ (y_k1_prime - (C @ mu_k1)))
	I = np.identity(2)
	sigma_k  = (I - K@C) @ sigma_k1

 	# Saving values
	a.append(mu_k[1][0])
	var_l.append(mu_k[1][0] - np.sqrt(sigma_k[1][1]))
	var_h.append(mu_k[1][0] + np.sqrt(sigma_k[1][1]))

# Plotting
gt = [-1 for i in range(len(a))]
plt.plot(a, 'r', label = "Estimated a", linewidth = 1)
plt.plot(gt, 'b', label="Ground truth", linewidth = 1)
plt.plot(var_l, 'g', label = "Variance low", linewidth = 1)
plt.plot(var_h, 'm', label = "Variance high", linewidth = 1)
plt.xlabel('Time step k')
plt.ylabel('a')
plt.title('Estimated a against time steps k')
plt.legend()

plt.savefig('ekf_plots.png')