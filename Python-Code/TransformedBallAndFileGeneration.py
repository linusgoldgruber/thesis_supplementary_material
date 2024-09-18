import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
np.set_printoptions(suppress=True, formatter={'float': lambda x: "{:.10f}".format(x)})

# CALCULATION
def sigmoid(value, a, b):

    # Ensure the value falls within the range [x, y]
    if a > b:
        raise ValueError("Lower bound x must be less than upper bound y")

    # Normalize the input value to the range [-6, 6] based on the interval [x, y]
    mid_point = (a + b) / 2
    range_width = (b - a) / 2

    if range_width == 0:
        raise ValueError("x and y must not be the same value")

    value_normalized = (value - mid_point) / range_width

    # Apply the standard sigmoid function    standard_sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_value = 1 / (1 + np.exp(-value_normalized))

    return sigmoid_value
def print_std_dev(parameters_array, time_step):
    std_dev = np.std(parameters_array[:, 3])
    print("Standard deviation of parameters_array[:, 3]:", std_dev, "at iteration", time_step)
def calculate_avg_distance_to_radius(num_points, radius):
    # Generate sphere points
    sphere_points = generate_sphere_points(num_points, radius)

    # Calculate distances between each pair of points
    distances = []
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = np.linalg.norm(sphere_points[i] - sphere_points[j])
            distances.append(distance)

    # Calculate average distance
    avg_distance = np.mean(distances)

    # Calculate percentage in comparison to the radius
    percentage = (avg_distance / radius) * 100
def transform_coordinates(coords, a, b):
    # Convert to numpy array for easier manipulation
    coords = np.array(coords)

    # Normalize coordinates from [-1, 1] to [0, 1]
    normalized_coords = (coords + 1) / 2

    # Scale normalized coordinates to [a, b]
    scaled_coords = normalized_coords * (b - a) + a

    return scaled_coords.tolist()

# GENERATION
def generate_custom_colormap(num_colors):
    # Generate a list of evenly spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)

    # Create a custom colormap with distinct colors
    colors = plt.cm.tab20(values)

    # Create a colormap object
    cmap = mcolors.ListedColormap(colors)

    return cmap
def generate_sphere_points(num_points, radius, noise_level=0.01, lower_bound=2, upper_bound=18):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        r = radius * np.sqrt(1 - y * y)  # scaled radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        # Add noise to each coordinate
        x += np.random.normal(0, noise_level)
        y += np.random.normal(0, noise_level)
        z += np.random.normal(0, noise_level)

        points.append([x, y, z])

    # Transform points to the new range
    points = transform_coordinates(points, lower_bound, upper_bound)
    return np.array(points)
def generate_ball_points(num_points, radius, noise_level=0.01, lower_bound=2, upper_bound=18):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        r = radius * (i / float(num_points - 1))**(1/3)  # distribute radius uniformly
        theta = phi * i
        z = 1 - (i / float(num_points - 1)) * 2
        r_xy = np.sqrt(1 - z * z)

        x = r * np.cos(theta) * r_xy
        y = r * np.sin(theta) * r_xy
        z = r * z + 0.22

        # Add noise to each coordinate if specified
        x += np.random.normal(0, noise_level)
        y += np.random.normal(0, noise_level)
        z += np.random.normal(0, noise_level)

        points.append([x, y, z])

    # Transform points to the new range
    points = transform_coordinates(points, lower_bound, upper_bound)
    return np.array(points)

# SIMULATION
def simulate_time_step(num_points, time_steps, diffusion_rate_MG, morphogen_secretion_coef, tfactor_coef, points_array, parameters_array, inhibition_constant):
    for iteration in range(time_steps):
        # Create a separate array to store the updated parameters
        updated_parameters_array = np.copy(parameters_array)

        # INTRACELLULAR:
        for cell_index in range(len(points_array)):
            # Get the cell coordinates
            cell = points_array[cell_index]

            # Calculate secretion amount: Product of coefficient and random value at index 0
            mg_secretion_amount_cell = morphogen_secretion_coef * parameters_array[cell_index, 0]


            # INTERCELLULAR: Apply morphogen diffusion to surrounding cells for each cell
            for target_cell_index in range(len(points_array)):
                #if cell_index != target_cell_index:  # Skip self
                if cell_index != target_cell_index:  # Skip self

                    target_cell = points_array[target_cell_index]               # Get the target cell coordinates
                    distance = np.linalg.norm(cell - target_cell)               # Calculate distance between cells
                    diffusion_factor = np.exp(-(diffusion_rate_MG * distance * distance))    # Calculate diffusion factor

                    # Add to morphogen concentration based on diffusion factor
                    updated_parameters_array[target_cell_index, 3]  += mg_secretion_amount_cell * diffusion_factor

        # Update parameters_array with the calculated values after all calculations for the time step
        parameters_array = np.copy(updated_parameters_array)

        # Print the standard deviation to check the mechanism
        print_std_dev(parameters_array, iteration)

        # Print parameters_array
        for index, params in enumerate(parameters_array):
            if parameters_array[index, 3] >= 0.01:
                print("Index ", index, ":", params)

        print("##############################################################")

        # Save parameters_array to a CSV file for each iteration
        #np.savetxt(f'parameters_array_{iteration}.csv', parameters_array[:, :4], delimiter=',', fmt='%.4f')
        np.savetxt(f'points_array{iteration+time_steps}.csv', points_array[:, :3], delimiter=',', fmt='%.4f')

    return parameters_array

# VISUALIZATION
def visualize_as_chart(csv_file):
    parameters_array = np.genfromtxt(csv_file, delimiter=',')
    num_points = len(parameters_array)
    indices = np.arange(num_points)
    values = parameters_array[:, 3]

    plt.bar(indices, values)
    plt.xlabel('Point Index')
    plt.ylabel('Value of parameters_array[:, 3]')
    plt.title('Values of parameters_array[:, 3] for Each Point')
    plt.show()
def visualize_from_csv(csv_file_prefix, start_iteration, end_iteration, num_visualizations, parameter):
    # Determine the total number of frames available
    total_frames = end_iteration - start_iteration + 1

    # Determine the step size between frames to visualize
    step_size = max(total_frames // num_visualizations, 1)

    # Load and visualize the frames according to the specified parameters
    for i in range(start_iteration, end_iteration + 1, step_size):
        parameters_array = np.genfromtxt(f'{csv_file_prefix}_{i}.csv', delimiter=',')
        num_points = len(parameters_array)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Normalize the color values for better visualization
        colors = parameters_array[:, parameter] / np.max(parameters_array[:, parameter])

        # determine the cell with the highest morphogen concentration
        max_position = np.argmax(parameters_array[:, parameter])

        # Mask to exclude the red dot from the main scatter plot loop
        mask = np.ones(num_points, dtype=bool)
        mask[max_position] = False

        # Plot all points except the one with the highest value
        ax.scatter(points_array[mask, 0], points_array[mask, 1], points_array[mask, 2], c=colors[mask], cmap='viridis')

        # Plot the point with the highest value in red
        ax.scatter(points_array[max_position, 0], points_array[max_position, 1], points_array[max_position, 2], c='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f'Sphere Points At Iteration {i}')
        plt.show()
def visualize_parameter_over_time(csv_file_prefix, start_iteration, end_iteration, num_visualizations, parameter_index):
    # Determine the total number of frames available
    total_frames = end_iteration - start_iteration + 1

    # Determine the step size between frames to visualize
    step_size = max(total_frames // num_visualizations, 1)

    # Generate a custom colormap with more colors
    num_colors = 80
    cmap = generate_custom_colormap(num_colors)

    fig, ax = plt.subplots()

    # Load and visualize the frames according to the specified parameters
    for i in range(start_iteration, end_iteration + 1, step_size):
        csv_file = f'{csv_file_prefix}_{i}.csv'
        parameters_array = np.genfromtxt(csv_file, delimiter=',')
        values = parameters_array[:, parameter_index]
        ax.plot(values, label=f'Time Step {i}', color=cmap((i - start_iteration) // step_size % num_colors))

    ax.set_xlabel('Point Index')
    ax.set_ylabel(f'Parameter {parameter_index}')
    ax.set_title(f'Development of Parameter {parameter_index} Over Time')
    ax.legend()
    plt.show()
def visualize_single_point_over_time(csv_file_prefix, point_index, parameter_index, start_iteration, end_iteration):
    # Determine the total number of frames available
    total_frames = end_iteration - start_iteration + 1

    # Generate a custom colormap with more colors
    num_colors = 80
    values = np.linspace(0, 1, num_colors)
    colors = plt.cm.tab20(values)
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots()

    time_steps = []
    parameter_values = []

    # Load and visualize the data for the specific point over time
    for i in range(start_iteration, end_iteration + 1):
        csv_file = f'{csv_file_prefix}_{i}.csv'
        parameters_array = np.genfromtxt(csv_file, delimiter=',')

        # Store the time step and the parameter value for the specific point
        time_steps.append(i)
        parameter_values.append(parameters_array[point_index, parameter_index])

    ax.plot(time_steps, parameter_values, label=f'Point {point_index} Parameter {parameter_index}', color=cmap(0))

    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Parameter {parameter_index}')
    ax.set_title(f'Change of Parameter {parameter_index} Over Time for Point {point_index}')
    ax.legend()
    plt.show()
def visualize_te_icm(parameters_array, points_array, num_points_TE, num_points_ICM):

    # Calculate the range of values for parameter 3 and define thresholds
    parameter_values = parameters_array[:, 3]
    parameter_min = np.min(parameter_values)
    parameter_max = np.max(parameter_values)
    parameter_range = parameter_max - parameter_min
    thresholds = np.linspace(0.10, 0.80, 70) * parameter_range + parameter_min

    best_threshold = None
    min_diff = float('inf')
    best_te_points = None
    best_icm_points = None

    # Iterate over each threshold to find the best one
    for threshold in thresholds:
        # Group points into TE and ICM based on the threshold
        te_indices = parameter_values < threshold
        icm_indices = parameter_values >= threshold

        te_count = np.sum(te_indices)
        icm_count = np.sum(icm_indices)

        diff = abs(te_count - num_points_TE) + abs(icm_count - num_points_ICM)

        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            best_te_points = points_array[te_indices]
            best_icm_points = points_array[icm_indices]

    print(f'Optimal Threshold: {best_threshold}')
    print(f'Optimal Threshold in percent: {best_threshold/parameter_range}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot TE points in one color
    ax.scatter(best_te_points[:, 0], best_te_points[:, 1], best_te_points[:, 2], c='blue', label='TE')

    # Plot ICM points in another color
    ax.scatter(best_icm_points[:, 0], best_icm_points[:, 1], best_icm_points[:, 2], c='green', label='ICM')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of TE and ICM Points')
    ax.legend()

    plt.show()

################################################### MAIN ##################################################################################################################

# Set Simulation Parameters
time_steps                      = 300
amount_of_parameters_per_point  =  5
blastocyst_radius               =  1

# Set Visualization Parameters
begin_visualization_at_step     =   0
end_visualization_at_step       =  300
num_visualizations              =  20
parameter_to_vizualise_sphere   =   3
parameter_to_vizualise_chart    =   3

# Set Diffusion Parameters
diffusion_rate_MG               = 0.06     # regulates the radius of diffusion (the higher, the smaller the radius) UNIT CIRCLE: 6
morphogen_coef                  = 1     # UNIT CIRCLE: 1
tfactor_coef                    = 2     # UNIT CIRCLE: 0.2
inhibition_constant             = 10    # "just" changes the baseline closely above zero that TF values return to after polarization,  UNIT CIRCLE: 10

# Generate points on the sphere surface
num_points_TE = 21
num_points_ICM = 19
num_points = num_points_TE + num_points_ICM  # Number of points on the sphere
points_array = generate_sphere_points(num_points_TE, blastocyst_radius)
points_array = np.concatenate((points_array, generate_ball_points(num_points_ICM, blastocyst_radius * 0.8)), axis=0)

# Create a new array with dimensions num_points * 4
parameters_array = np.zeros((num_points, amount_of_parameters_per_point))

# Set your desired parameters
for i in range(num_points):
    parameters_array[i] = [np.random.uniform(1, 1.2),   # First  parameter: Random between 1 and 1.2, Stochastic factor
                           0,                           # Second parameter: -
                           0,                           # Third  parameter: -
                           0,                           # Fourth parameter: MG-Concentration in the cell
                           0]                           # Fifth  parameter: -

################################################### Simulate ###############################################################################################################
parameters_array = simulate_time_step(num_points, time_steps, diffusion_rate_MG, morphogen_coef, tfactor_coef, points_array, parameters_array, inhibition_constant)

################################################### Visualization ###########################################################################################################
calculate_avg_distance_to_radius(num_points_TE, blastocyst_radius)


# SINGLE PARAMETER for ALL POINTS over time
visualize_parameter_over_time('parameters_array', begin_visualization_at_step, end_visualization_at_step - 1, num_visualizations, parameter_to_vizualise_chart)

# Visualize the TE and ICM points
visualize_te_icm(parameters_array, points_array, num_points_TE, num_points_ICM)


# Print the new coordinates
for coord in points_array:
    print(coord)

# Line to append after each coordinate line
additional_line = (
    " \t0.000000,0.000000,0.000000,\t 0, 4278203136,, \t4294967295, 0.000000, 0.000000, 0.000000, "
    "0.000000, 4294967295, -nan, 0.000000, 0, \t4294967295, 0.000000, 0.000000, 0.000000, "
    "0.000000, 4294967295, -nan, 0.000000, 0, \t4294967295, 0.000000, 0.000000, 0.000000, "
    "0.000000, 4294967295, -nan, 0.000000, 0, \t4294967295, 0.000000, 0.000000, 0.000000, "
    "0.000000, 4294967295, -nan, 0.000000, 0, \t1, 65025, 133, 0, \t\t4294967295, 4294967295,, "
    "4294967295, 4294967295,, 4294967295, 4294967295,, 4294967295, 4294967295,, 0.000000, "
    "0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, "
    "0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, \t2147483647, 0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0\t"
)

# Add index, double comma, and additional line
indexed_coordinates = []
for i, coord in enumerate(points_array):
    coord_formatted = [f"{num:.6f}" for num in coord]
    line = [i, ''] + coord_formatted + additional_line.split(',')
    indexed_coordinates.append(line)

# Write to a new CSV file
with open('indexed_coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerows(indexed_coordinates)
