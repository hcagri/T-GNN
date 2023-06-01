import numpy as np

# Generate random pedestrian positions for testing
np.random.seed(42)  # Set seed for reproducibility
N = 5  # Number of pedestrians
positions = np.random.rand(N, 2, 20)  # Random positions for 20 seconds

# Calculate the mean x and y locations at the 8th second
locations_8th_sec = positions[:, :, 7]
mean_x = np.mean(locations_8th_sec[:, 0])
mean_y = np.mean(locations_8th_sec[:, 1])

mean_position = np.array([mean_x, mean_y])
mean_position = np.reshape(mean_position, (1, 2, 1))

# Subtract the mean position from all the pedestrian positions
positions_shifted = positions - mean_position

# Print the original positions
print("Original positions:")
print(positions)
print()

# Print the mean x and y locations
print("Mean x location:", mean_x)
print("Mean y location:", mean_y)
print()

# Print the shifted positions
print("Shifted positions:")
print(positions_shifted)

