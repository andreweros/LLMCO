import numpy as np
import matplotlib.pyplot as plt
from .floor_plan_util import Building

def calculate_coverage_on_pathloss_result(files, building_name, threshold=100):

    # Create a building object
    building = Building(building_name)

    # Set up pathloss map at the same size of building and fill it with 150
    pathloss_map = np.full((int(building.building_details['Height']), int(building.building_details['Width'])), 150)
    coverage_map = np.zeros_like(pathloss_map)

    for file in files:
        # TODO: Here you should use your own implementation (ray tracer) to get pathloss data
         pathloss_tmp = building.get_pathloss_data(file)
         # compare pathloss_tmp with pathloss, and keep the lowest pathloss
         pathloss_map = np.minimum(pathloss_map, pathloss_tmp)

    # Coverage map is a binary map, 1 for covered, 0 for not covered
    coverage_map[pathloss_map < threshold] = 1

    # Calculate the coverage percentage
    coverage = np.mean(coverage_map) * 100

    boundary_info = building.building_details

    # Reduce the boundary info by 20%
    for key in boundary_info:
        boundary_info[key] = float(boundary_info[key]) * 0.9

    # Show pathloss map with color bar (0-150)
    plt.imshow(pathloss_map, cmap='jet', vmin=0, vmax=150)
    plt.title(f"Pathloss Map")
    plt.colorbar()
    plt.show()

    return coverage, boundary_info, pathloss_map, coverage_map


