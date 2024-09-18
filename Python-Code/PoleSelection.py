import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
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

# GENERATION
def generate_custom_colormap(num_colors):
    # Generate a list of evenly spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)

    # Create a custom colormap with distinct colors
    colors = plt.cm.tab20(values)

    # Create a colormap object
    cmap = mcolors.ListedColormap(colors)

    return cmap
def generate_sphere_points(num_points, radius, noise_level=0.07):
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

    return np.array(points)
def generate_ball_points(num_points, radius, noise_level=0.01):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        # Calculate a uniform distribution of points within the sphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)

        theta = phi * i
        z = 1 - (i / float(num_points - 1)) * 2
        r_xy = np.sqrt(1 - z * z)

        # Convert u, v to spherical coordinates
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        # Add noise to each coordinate if specified
        x += np.random.normal(0, noise_level)
        y += np.random.normal(0, noise_level)
        z += np.random.normal(0, noise_level)

        points.append([x, y, z])

    return np.array(points)

# SIMULATION
def simulate_time_step(time_steps, diffusion_rate_MG, morphogen_secretion_coef, tfactor_coef, points_array, parameters_array, inhibition_constant):
    for iteration in range(time_steps):

        # Create a separate array to store the updated parameters
        updated_parameters_array = np.copy(parameters_array)

        # INTRACELLULAR:
        for cell_index in range(len(points_array)):
            # Get the cell coordinates
            cell_location = points_array[cell_index]

        # Calculate secretion amounts

            # Product of coefficient, random value at index 0 and intracellular MG-concentration, IF present (avoid multiplication with 0)
            mg_secretion_amount_cell = morphogen_secretion_coef * parameters_array[cell_index, 0]
            if sigmoid(parameters_array[cell_index, 3], 0, 100)  != 0:
                mg_secretion_amount_cell *= sigmoid(parameters_array[cell_index, 3], 0, 100)

            # Calculated to be a product of the coefficient and the random value at index 1
            tfactor_amount_cell = tfactor_coef * parameters_array[cell_index, 1]

            # Increase TF Concentration depending on genetics and morphogen concentration
            updated_parameters_array[cell_index, 2] += tfactor_amount_cell

            # Reduce TF Concentration by a factor between 0 and 1, depending on morphogen presence and concentration
            if parameters_array[cell_index, 3] != 0:
                updated_parameters_array[cell_index, 2] *= parameters_array[cell_index, 3] / inhibition_constant

            # INTERCELLULAR: Apply morphogen diffusion to surrounding cells for each cell
            for target_cell_index in range(len(points_array)):
                if cell_index != target_cell_index:  # Skip self
                    target_cell_location = points_array[target_cell_index]               # Get the target cell coordinates
                    distance = np.linalg.norm(cell_location - target_cell_location)               # Calculate distance between cells
                    diffusion_factor = np.exp(-diffusion_rate_MG * distance)    # Calculate diffusion factor

                    # Add morphogen to other cells
                    #if parameters_array[cell_index, 4] == 1 and parameters_array[target_cell_index, 4] != 1:
                        # Add to morphogen concentration based on diffusion factor
                    updated_parameters_array[target_cell_index, 3] += mg_secretion_amount_cell * diffusion_factor
                    print(f"Updated parameters from cell {cell_index} for target cell {target_cell_index}: {updated_parameters_array[target_cell_index, 3]} + {mg_secretion_amount_cell} * {diffusion_factor}") if target_cell_index == 10 else None


                # If pole parameter 4 is activated for a cell, print that to the console
                if updated_parameters_array[cell_index, 4] == 1 and parameters_array[cell_index, 4] == 0:
                    print("POLE UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED UPDATED")

                if parameters_array[cell_index, 2] > 800:
                    #print("\n\n\n\nITERATION = TIME STEP\n\n\n\n")
                    parameters_array[cell_index, 4] = 1
                    return

        # Update parameters_array with the calculated values after all calculations for the time step
        parameters_array = np.copy(updated_parameters_array)

        # Print the standard deviation to check the mechanism
        print_std_dev(parameters_array, iteration)

        # Print parameters_array
        for index, params in enumerate(parameters_array):
            if parameters_array[index, 2] >= 0.01:
                print("Index ", index, ":", params)

        # Print the position of the item with the highest value in position 3 after the last time step
        max_position = np.argmax(parameters_array[:, 3])
        # print("Position of item with the highest value in position 3:", max_position)

        print("##############################################################")

        # Save parameters_array to a CSV file for each iteration
        np.savetxt(f'parameters_array_{iteration}.csv', parameters_array[:, :4], delimiter=',', fmt='%.4f')

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
        ax.scatter(points_array[mask, 0], points_array[mask, 1], points_array[mask, 2], c=colors[mask], cmap='viridis', s=100)

        # Plot the point with the highest value in red
        ax.scatter(points_array[max_position, 0], points_array[max_position, 1], points_array[max_position, 2], c='red', s=100)

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

################################################### MAIN ##################################################################################################################

# Set Simulation Parameters
time_steps                      = 50
amount_of_parameters_per_point  =  5
blastocyst_radius               =  1

# Set Visualization Parameters
begin_visualization_at_step     =  0
end_visualization_at_step       = 50
num_visualizations              = 20
parameter_to_vizualise_sphere   =  2
parameter_to_vizualise_chart    =  2

# Set Diffusion Parameters
diffusion_rate_MG               = 2  # regulates the number of poles being created: the smaller, the more local maxima
morphogen_coef                  = 0.2
tfactor_coef                    = 0.2
inhibition_constant             = 5    # "just" changes the baseline closely above zero that TF values return to after polarization

# Generate points on the sphere surface
num_points = 20  # Number of points on the sphere
points_array = generate_sphere_points(num_points, blastocyst_radius)


# Create a new array with dimensions num_points * 4
parameters_array = np.zeros((num_points, amount_of_parameters_per_point))

# Set your desired parameters
for i in range(num_points):
    parameters_array[i] = [np.random.uniform(1, 2),     # First  parameter: Random between 1 and 2: Receptor affinity
                           np.random.uniform(1, 2),     # Second parameter: Random between 1 and 2: Receptor affinity
                           0,                           # Third  parameter: TF-Concentraion in the cell
                           0,                           # Fourth parameter: MG-Concentration in the cell
                           0]                           # Fifth  parameter: -


################################################### Simulate ###############################################################################################################
parameters_array = simulate_time_step(
                        time_steps,
                        diffusion_rate_MG,
                        morphogen_coef,
                        tfactor_coef,
                        points_array,
                        parameters_array,
                        inhibition_constant
                        )

################################################### Visualization ###########################################################################################################
# SPHERE with COLOURED PARAMETER from CSV files
visualize_from_csv('parameters_array', begin_visualization_at_step, end_visualization_at_step - 1, num_visualizations, parameter_to_vizualise_sphere)

# BAR CHART from a chosen .csv file
#csv_file = 'parameters_array_90.csv'  # Replace with the path to your .csv file
#visualize_as_chart(csv_file)

# SINGLE PARAMETER for ALL POINTS over time
visualize_parameter_over_time('parameters_array', begin_visualization_at_step, end_visualization_at_step - 1, num_visualizations, parameter_to_vizualise_chart)

# SINGLE PARAMETER for SINGLE POINTS over time
point_index = np.argmax(parameters_array[:, 2]) # Index of the point to visualize
visualize_single_point_over_time('parameters_array', point_index, 3, begin_visualization_at_step, end_visualization_at_step - 1)



