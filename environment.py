import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_heatmap(width=10, height=10):
    # Create a 2D array of shape (height, width) with random values
    data = np.random.rand(height, width)

    # Normalize the data to the range [0, 1]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Initialize an RGB image
    rgb_image = np.zeros((height, width, 3))

    # Set RGB values based on the normalized data
    for y in range(height):
        for x in range(width):
            intensity = normalized_data[y, x]
            # Set blue to intensity and red to 1 - intensity
            rgb_image[y, x, 0] = (1 - intensity)  # Red channel
            rgb_image[y, x, 1] = 0                # Green channel
            rgb_image[y, x, 2] = intensity         # Blue channel

    return rgb_image

def rgb_to_vector_array(rgb_image):
    height, width, _ = rgb_image.shape

    # Initialize a list to store the vectors
    vector_array = []

    for y in range(height):
        for x in range(width):
            # Get the red and blue channel values
            r_value = rgb_image[y, x, 0]  # Red channel value in [0, 1]
            b_value = rgb_image[y, x, 2]  # Blue channel value in [0, 1]

            # Calculate zvel based on the average of the blue and red channels
            zvel = ((b_value * 255 + r_value * 255) / 2) - (b_value * 255)
            zvel = max(0, zvel)  # Ensure zvel is not negative
            
            # Randomly initialize xvel and yvel in the range [0, 2]
            xvel = np.random.uniform(0, 2)
            yvel = np.random.uniform(0, 2)

            # Create the vector [x, y, xvel, yvel, zvel]
            vector = [x, y, xvel, yvel, zvel]
            vector_array.append(vector)

    return np.array(vector_array)

# Create the heatmap
rgb_image = create_heatmap()

r_value_00 = rgb_image[0, 0, 0]  # Red channel
g_value_00 = rgb_image[0, 0, 1]  # Green channel
b_value_00 = rgb_image[0, 0, 2]  # Blue channel
print(f"RGB values at pixel (0, 0): R={r_value_00:.2f}, G={g_value_00:.2f}, B={b_value_00:.2f}")

# Convert the RGB image to a vector array
vector_array = rgb_to_vector_array(rgb_image)

# Display the shape of the vector array
print("Vector array shape:", vector_array.shape)
print("Sample vector:", vector_array[0])  # Print the first vector for verification


# Optionally, display the heatmap
plt.imshow(rgb_image)
plt.axis('off')  # Hide the axes
plt.title('2D Heatmap (Enhanced Blue to Red)')
plt.show()
