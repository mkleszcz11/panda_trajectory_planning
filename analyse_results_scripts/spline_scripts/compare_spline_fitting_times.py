import matplotlib.pyplot as plt
import numpy as np

# # Data - sim
# num_waypoints = np.array([6, 11, 16, 26, 36, 51, 76, 101, 151])
# cubic = np.array([0.408, 0.657, 0.747, 0.952, 0.980, 1.380, 1.799, 1.807, 2.695])
# quintic_bspline = np.array([2.582, 3.304, 3.346, 4.905, 4.888, 6.696, 6.915, 9.006, 11.381])
# quintic_poly = np.array([2.315, 2.796, 3.900, 4.688, 5.410, 5.643, 6.982, 9.461, 11.921])

# Data - real robot
num_waypoints = np.array([6, 11, 16, 26, 36, 51])
cubic = np.array([0.210, 0.267, 0.262, 0.323, 0.491, 0.459])
quintic_bspline = np.array([0.922, 1.029, 1.162, 1.457, 1.799, 2.185])
quintic_poly = np.array([0.864, 1.008, 1.121, 1.406, 1.736, 2.045])

sqrt_wp = np.sqrt(num_waypoints)  # Basis for linear regression

# Fit a*x^0.5 + b model using np.polyfit on sqrt(x)
coef_cubic = np.polyfit(sqrt_wp, cubic, 1)
coef_bspline = np.polyfit(sqrt_wp, quintic_bspline, 1)
coef_poly = np.polyfit(sqrt_wp, quintic_poly, 1)

# Evaluation range
x_fit = np.linspace(min(num_waypoints), max(num_waypoints), 300)
sqrt_x_fit = np.sqrt(x_fit)

fit_cubic = coef_cubic[0] * sqrt_x_fit + coef_cubic[1]
fit_bspline = coef_bspline[0] * sqrt_x_fit + coef_bspline[1]
fit_poly = coef_poly[0] * sqrt_x_fit + coef_poly[1]

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(num_waypoints, cubic, label='Cubic (data)', marker='o')
plt.scatter(num_waypoints, quintic_bspline, label='Quintic B-spline (data)', marker='s')
plt.scatter(num_waypoints, quintic_poly, label='Quintic Polynomial (data)', marker='^')

plt.plot(x_fit, fit_cubic, label='Cubic (sqrt fit)')
plt.plot(x_fit, fit_bspline, label='Quintic B-spline (sqrt fit)')
plt.plot(x_fit, fit_poly, label='Quintic Polynomial (sqrt fit)')

plt.xlabel('Number of Waypoints')
plt.ylabel('Fitting Time [s]')
plt.title('Spline Fitting Time vs. Number of Waypoints (Real Hardware)')
plt.grid(True)
plt.legend()
plt.tight_layout()
# save to the file with high resolution
plt.savefig('spline_fitting_time_vs_waypoints_real_robot.png', dpi=300)
plt.show()

