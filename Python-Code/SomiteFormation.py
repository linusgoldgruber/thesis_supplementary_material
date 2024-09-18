
import numpy as np
import pandas as pd
import pyvista as pv  # Import PyVista
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
np.set_printoptions(suppress=True, formatter={'float': lambda x: "{:.5f}".format(x)})

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
    print("Standard deviation of CLOCK (should oscillate):", std_dev, "at iteration", time_step)
def update_CLOCK_clock(cell_index, parameters_array, TEMP_parameters_array, coef_CLOCK):
    secretion_amount_old    =      parameters_array[cell_index, 4]
    secretion_amount        = TEMP_parameters_array[cell_index, 4]
    secretion_coef          =      parameters_array[cell_index, 0]
    #concentration_old      =      parameters_array[cell_index, 3]
    concentration           = TEMP_parameters_array[cell_index, 3]
    activated               =      parameters_array[cell_index, 7]
    #is_activated_new       = TEMP_parameters_array[cell_index, 7]

    if parameters_array[cell_index, 1] == 0:    # Only calculate for non-somite cells

        if 0 < concentration <= 80 and not activated:  # If the cell should increase the amount of secretion of CLOCK

            print("ADDING: ", cell_index, ", Concentration went from", parameters_array[cell_index, 3], " to ", TEMP_parameters_array[cell_index, 3], ".") if cell_index == 300 else None
            TEMP_parameters_array[cell_index, 4] =  secretion_amount_old + secretion_coef * 0.1 * coef_CLOCK # Cell increases its secretion

        elif concentration >= 80 and not activated:  # If the cell hits the threshold

            TEMP_parameters_array[cell_index, 4] = 0                                                    # Cell stops secreting
            TEMP_parameters_array[cell_index, 3] -= concentration * 0.3                                 # Cell reduces its CLOCK-concentration
            TEMP_parameters_array[cell_index, 7] = 1                                                    # Cell is marked as having hit the threshold

        elif 20 <= concentration and activated:  # If the cell was activated and should decrease the output

            TEMP_parameters_array[cell_index, 4] = 0                                                    # Cell resets Secretion
            TEMP_parameters_array[cell_index, 3] -= concentration * 0.5 if concentration > 1 else 0    # Cell further reduces its CLOCK-concentration
            print("REDUCING: ", cell_index, ", Concentration went from", parameters_array[cell_index, 3], " to ", TEMP_parameters_array[cell_index, 3], ".") if cell_index == 300 else None
            TEMP_parameters_array[cell_index, 7] = 1
        else:
            TEMP_parameters_array[cell_index, 7] = 0    # Cell is deactivated, i.e. threshold hit gets reset
            TEMP_parameters_array[cell_index, 4] = 0    # Cell resets its CLOCK-Secretion
            TEMP_parameters_array[cell_index, 3] = 0    # Cell resets its CLOCK-Concentration
    else:  # reset somite cells
        TEMP_parameters_array[cell_index, 7] = 0    # Cell is deactivated, i.e. threshold hit gets reset
        TEMP_parameters_array[cell_index, 4] = 0    # Cell resets its CLOCK-Secretion
        TEMP_parameters_array[cell_index, 3] = 0    # Cell resets its CLOCK-Concentration
def update_WAVE_wave(cell_index, coef_WAVE, parameters_array, TEMP_parameters_array, iteration, time_steps):
    pole                    = TEMP_parameters_array[cell_index, 2]

    if pole:
        TEMP_parameters_array[cell_index, 6] = 100 - (iteration / time_steps) * 100
        #print("POLE! Value: ", TEMP_parameters_array[cell_index, 6])

def calc_pole_and_somites(points_array, TEMP_parameters_array):
    # Create Pole for CLOCK Clock
    # Find the index of the point with the highest z-value
    highest_z_index = np.argmax(points_array[:, 2])

    # Update the value of parameters_array at the specified cell_index
    TEMP_parameters_array[highest_z_index, 2] = 1
    TEMP_parameters_array[highest_z_index, 3] = 1
    # Calculate the number of points that correspond to the bottom 5%
    num_bottom_points = int(0.05 * num_points)

    # Get the indices of the points with the lowest z values
    bottom_indices = np.argsort(points_array[:, 2])[:num_bottom_points]

    for i in bottom_indices:
        TEMP_parameters_array[i, 1] = 1
        print("Somite: ", i)


    return highest_z_index
def calculate_average_distance(points_array):  # Calculate average distance between points
    total_distance = 0
    num_distances = 0
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            total_distance += np.linalg.norm(points_array[i] - points_array[j])
            num_distances += 1
    return total_distance / num_distances

# GENERATION
def generate_custom_colormap(num_colors):
    # Generate a list of evenly spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)

    # Create a custom colormap with distinct colors
    colors = plt.cm.tab20(values)

    # Create a colormap object
    cmap = mcolors.ListedColormap(colors)

    return cmap
def generate_presomitic_mesoderm_points(num_points, bow_radius, tube_radius, length, noise_level=0.01):

    def generate_cylinder_points(num_points, tube_radius, length):
        """Generate points within a cylinder."""
        points = np.zeros((num_points, 3))
        for i in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            r = tube_radius * np.sqrt(np.random.uniform(0, 1))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(0, length)
            points[i] = [x, y, z]
        return points

    def generate_bow_points(num_points, bow_radius, tube_radius, length):
        """Generate points within a bow shape."""
        points = np.zeros((num_points, 3))
        for i in range(num_points):
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)  # Bow arc segment
            x = bow_radius * np.sin(theta) + np.random.uniform(-0.2 * bow_radius, 0.2 * bow_radius)
            y = np.random.uniform(-tube_radius , tube_radius)
            z = bow_radius * np.cos(theta) + length * 0.9 + np.random.uniform(0, 0.5 * bow_radius)
            points[i] = [x, y, z]
        return points

    # Split points between cylinders and bow, ensuring the total matches num_points
    num_cylinder_points = int(num_points * 0.75)
    num_bow_points = num_points - num_cylinder_points

    # Generate points for the first cylinder
    cylinder1_points = generate_cylinder_points(num_cylinder_points // 2, tube_radius, length)
    # Translate the first cylinder points to align with the bow
    cylinder1_points[:, 0] -= bow_radius

    # Generate points for the second cylinder
    cylinder2_points = generate_cylinder_points(num_cylinder_points // 2, tube_radius, length)
    # Translate the second cylinder points to align with the bow
    cylinder2_points[:, 0] += bow_radius

    # Generate points for the bow
    bow_points = generate_bow_points(num_bow_points, bow_radius, tube_radius, length)

    # Concatenate all points
    points_array = np.vstack((cylinder1_points, cylinder2_points, bow_points))

    # Add noise to the points
    noise = noise_level * np.random.randn(points_array.shape[0], 3)
    points_array += noise

    # Adjust the total number of points if necessary
    while len(points_array) < num_points:
        print("Warning 1: length of array points_array is smaller than num_points.")
        extra_points = generate_cylinder_points(1, tube_radius, length)
        extra_points[:, 0] -= bow_radius
        points_array = np.vstack((points_array, extra_points))

    while len(points_array) > num_points:
        print("Warning 2:  length of array points_array is greater than num_points.")
        points_array = points_array[:-1]

    print("Points Array length: ", len(points_array))

    return points_array

# SIMULATION
def simulate_time_step(time_steps, diffusion_rate_CLOCK, diffusion_rate_WAVE, coef_CLOCK, coef_WAVE, points_array, parameters_array):
    # Create a separate array to store the updated parameters
    TEMP_parameters_array = np.copy(parameters_array)
    # Calculate pole position and somite-indices
    highest_z_index = calc_pole_and_somites(points_array, TEMP_parameters_array)
    somite_count  = 0
    print("POLE: ", highest_z_index)

    # Calculate average distance between points
    average_distance = calculate_average_distance(points_array)
    threshold_distance = 0.25 * average_distance

    for iteration in range(time_steps):
        somite_add    = 0


        ### INTRACELLULAR: START
        for cell_index in range(len(points_array)):
            # Get the cell coordinates
            cell = points_array[cell_index]

            # If the cell got activated and should increase the amount of diffusion
            update_CLOCK_clock(cell_index, parameters_array, TEMP_parameters_array, coef_CLOCK)
            update_WAVE_wave(cell_index, coef_WAVE, parameters_array, TEMP_parameters_array, iteration, time_steps)


            ### INTERCELLULAR: Apply morphogen diffusion to surrounding cells for each cell
            for target_cell_index in range(len(points_array)):
                #if cell_index != target_cell_index:  # Skip self
                if cell_index != target_cell_index:  # Skip self

                    target_cell = points_array[target_cell_index]               # Get the target cell coordinates
                    distance = np.linalg.norm(cell - target_cell)               # Calculate distance between cells

                    ### CLOCK
                    diffusion_factor_CLOCK = np.exp(-diffusion_rate_CLOCK * distance ** 4) if distance < threshold_distance else 0  # Calculate diffusion factor

                    if parameters_array[target_cell_index, 1] == 0 and parameters_array[target_cell_index, 7] == 0:  # Add to morphogen concentration based on diffusion factor
                        TEMP_parameters_array[target_cell_index, 3]  += TEMP_parameters_array[cell_index, 4] * diffusion_factor_CLOCK

                    ### WAVE
                    diffusion_factor_WAVE = np.exp(-diffusion_rate_WAVE * 0.65 * distance)  # if distance < threshold_distance else 0   # Calculate diffusion factor

                    if cell_index == highest_z_index and TEMP_parameters_array[target_cell_index, 1] == 0:
                        TEMP_parameters_array[target_cell_index, 5]  = TEMP_parameters_array[cell_index, 6] * diffusion_factor_WAVE
                    if cell_index == highest_z_index and TEMP_parameters_array[target_cell_index, 1] != 0:
                        TEMP_parameters_array[target_cell_index, 5]  = 0

            # Turn presomitic cells into somite cells
            if TEMP_parameters_array[cell_index, 3] > 70 and TEMP_parameters_array[cell_index, 5] < 20 and TEMP_parameters_array[cell_index, 1] == 0:
                TEMP_parameters_array[cell_index, 1] = 1
            if TEMP_parameters_array[cell_index, 1] > 0 and parameters_array[cell_index, 1] == 0:
                somite_add += 1
        ### INTRACELLULAR: END

        print("SOMITE ADD: ", somite_add, "\n")
        somite_count += somite_add
        print("SOMITE COUNT: ", somite_count, ", +", somite_add, "\n")

        parameters_array = np.copy(TEMP_parameters_array)

        create_csvs(parameters_array, iteration)


    return parameters_array

# VISUALIZATION
def create_csvs(parameters_array, iteration):
    # Print the standard deviation to check the mechanism
    print_std_dev(parameters_array, iteration)

    # Print parameters_array
    """for index, params in enumerate(parameters_array):
        #if parameters_array[index, 3] >= 0.01:
        print("Index ", index, ":", params)"""
    print("##############################################################")

    # Save parameters_array to a CSV file for each iteration
    np.savetxt(f'parameters_array_{iteration}.csv', parameters_array[:len(parameters_array), :8], delimiter=',', fmt='%.4f')

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
def visualize_from_csv(csv_file_prefix, start_iteration, end_iteration, num_visualizations, parameter, points_array):
    # Determine the total number of frames available
    total_frames = end_iteration - start_iteration + 1

    # Determine the step size between frames to visualize
    step_size = max(total_frames // num_visualizations, 1)

    # Load and visualize the frames according to the specified parameters
    for i in range(start_iteration, end_iteration, step_size):
        parameters_array = np.genfromtxt(f'{csv_file_prefix}_{i}.csv', delimiter=',')
        print("Parameters Array Length: ", len(parameters_array))
        print(parameters_array)
        print("Points Array Length: ", len(points_array))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Normalize the color values for better visualization
        colors = parameters_array[:, parameter] / np.max(parameters_array[:, parameter])

        # Plot all points with color-coding based on the parameter value
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c=colors, cmap='viridis')

        # Set equal scaling for all axes
        max_range = np.array([points_array[:, 0].max() - points_array[:, 0].min(),
                              points_array[:, 1].max() - points_array[:, 1].min(),
                              points_array[:, 2].max() - points_array[:, 2].min()]).max() / 2.0

        mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
        mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
        mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f'Parameter {parameter}, Iteration {i}, Max: {np.max(parameters_array[:, parameter])} & Min: {np.min(parameters_array[:, parameter])}')
        plt.show()
def visualize_from_csv_pyvista(csv_file_prefix, start_iteration, end_iteration, num_visualizations, parameter, points_array, window_size=[1600, 1200]):
    total_frames = end_iteration - start_iteration + 1
    step_size = max(total_frames // num_visualizations, 1)

    # Preload all parameters arrays into memory
    parameters_arrays = []
    for i in range(start_iteration, end_iteration, step_size):
        parameters_array = np.genfromtxt(f'{csv_file_prefix}_{i}.csv', delimiter=',')
        parameters_arrays.append(parameters_array)

    # Check if all parameters_arrays have the same length as points_array
    for parameters_array in parameters_arrays:
        if len(parameters_array) != len(points_array):
            print(f"Length mismatch: parameters_array has {len(parameters_array)} elements, points_array has {len(points_array)} elements.")
            return

    # Create a single plotter to be reused
    plotter = pv.Plotter(window_size=window_size)

    # Iterate through the pre-loaded data and visualize
    for idx, parameters_array in enumerate(parameters_arrays):

        # Normalize the color values for better visualization
        colors = parameters_array[:, parameter] / np.max(parameters_array[:, parameter])

        # Create a point cloud with PyVista
        point_cloud = pv.PolyData(points_array)
        point_cloud['Paramter Range'] = colors

        # Clear the previous plot and add the new point cloud
        plotter.clear()
        plotter.add_mesh(point_cloud, scalars='Paramter Range', cmap='viridis', point_size=20, render_points_as_spheres=True)

        # Set equal scaling for all axes
        max_range = np.array([points_array[:, 0].max() - points_array[:, 0].min(),
                              points_array[:, 1].max() - points_array[:, 1].min(),
                              points_array[:, 2].max() - points_array[:, 2].min()]).max() / 2.0

        mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
        mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
        mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5

        plotter.camera.SetPosition(mid_x, mid_y, mid_z + max_range * 2)
        plotter.camera.SetFocalPoint(mid_x, mid_y, mid_z)
        plotter.camera.SetViewUp(0, 1, 0)
        plotter.reset_camera_clipping_range()

        # Render the scene
        plotter.show(auto_close=False)

    # Close the plotter after all frames are visualized
    plotter.close()
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
def visualize_points_over_time(csv_file_prefix, point_indices, parameter_index, start_iteration, end_iteration):
    # Determine the total number of frames available
    total_frames = end_iteration - start_iteration + 1

    # Generate a custom colormap with more colors
    num_colors = 80
    values = np.linspace(0, 1, num_colors)
    colors = plt.cm.tab20(values)
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots()

    # Load and visualize the data for the range of points over time
    for point_index in point_indices:
        time_steps = []
        parameter_values = []

        for i in range(start_iteration, end_iteration + 1):
            csv_file = f'{csv_file_prefix}_{i}.csv'
            parameters_array = np.genfromtxt(csv_file, delimiter=',')

            # Store the time step and the parameter value for the specific point
            time_steps.append(i)
            parameter_values.append(parameters_array[point_index, parameter_index])

        ax.plot(time_steps, parameter_values, label=f'Point {point_index} Parameter {parameter_index}', color=cmap(point_index % num_colors))

    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Parameter {parameter_index}')
    ax.set_title(f'Change of Parameter {parameter_index} Over Time for Points {point_indices}')
    #ax.legend()
    plt.show()
################################################### MAIN ##################################################################################################################

# Set Simulation Parameters
time_steps                      =  70       # Number of iterative steps of the simulation
amount_of_parameters_per_point  =   8       # Needs to match number of parameters set at line 477

# Set Visualization Parameters
begin_visualization_at_step     =   0       # min: 0
end_visualization_at_step       =  69       # max: time_steps - 1
num_visualizations              =  25       # max: time_steps
parameter_to_vizualise          =   5       # used later on in visualisation functions
somites_bow_radius              =   3       # parameter for point generation
somites_tube_radius             =   0.5     # parameter for point generation
somites_length                  =  20       # parameter for point generation

# Set Diffusion Parameters
diffusion_rate_CLOCK              =   0.1     # regulates the radius of diffusion (the higher, the smaller the radius)
diffusion_rate_WAVE          =   0.1     # regulates the radius of diffusion (the higher, the smaller the radius)
coef_CLOCK                        =   1       # change wavelength of CLOCK-secretion
coef_WAVE                    =   0.25    # change the absolute amount if secretion of WAVE

# Generate points for the presomitic mesoderm
num_points = 500  # Number of cells to simulate
points_array = generate_presomitic_mesoderm_points(num_points, somites_bow_radius, somites_tube_radius, somites_length)

# Create a new array with dimensions num_points * 4
parameters_array = np.zeros((num_points, amount_of_parameters_per_point))

# Set your desired parameters
for i in range(num_points - 1):
    parameters_array[i] = [np.random.uniform(1, 1.03),  # First   parameter: Random between 1 and 1.2, Stochastic factor
                           0,                           # Second  parameter: bool Somite y/n
                           0,                           # Third   parameter: bool Pole y/n
                           0,                           # Fourth  parameter: CLOCK-Concentration
                           0,                           # Fifth   parameter: CLOCK-Output (for now only activ ein the pole)
                           0,                           # Sixth   parameter: WAVE-Concentration
                           0,                           # Seventh parameter: WAVE-Output
                           0]                           # Eighth  parameter: bool, if CLOCK is going up or down (if it hit the threshold or not)

# Simulate ###############################################################################################################
#p
# arameters_array = simulate_time_step(time_steps, diffusion_rate_CLOCK, diffusion_rate_WAVE, coef_CLOCK, coef_WAVE, points_array, parameters_array)

# Visualization ###########################################################################################################
visualize_from_csv('parameters_array', begin_visualization_at_step, end_visualization_at_step, num_visualizations, 5, points_array)
#visualize_from_csv('parameters_array', begin_visualization_at_step, end_visualization_at_step, num_visualizations, 1, points_array)
#visualize_from_csv_pyvista('parameters_array', begin_visualization_at_step, end_visualization_at_step, num_visualizations, 3, points_array)
#visualize_parameter_over_time('parameters_array', begin_visualization_at_step, end_visualization_at_step - 1, num_visualizations, parameter_to_vizualise)
#visualize_single_point_over_time('parameters_array', 30, 5, begin_visualization_at_step, end_visualization_at_step)

#visualize_single_point_over_time('parameters_array', 200, 5, begin_visualization_at_step, end_visualization_at_step)
point_indices = range(1, 450)
#visualize_points_over_time('parameters_array', point_indices, 5, begin_visualization_at_step, end_visualization_at_step - 1)
