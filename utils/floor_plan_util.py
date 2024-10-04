import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This class is used to get building details and image
class Building:
    def __init__(self, building_name):
        self.building_name = building_name
        self.building_details = self._get_building_detail()
        self.building_image_path = f"building_images/{building_name}.png"
    def _get_building_detail(self):
        df = pd.read_csv(f"building_details/{self.building_name}_detail.csv")
        building_detail_dict = {'X_min': df['X_min'].values[0], 'X_max': df['X_max'].values[0],
                                'Y_min': df['Y_min'].values[0], 'Y_max': df['Y_max'].values[0],
                                'Width': df['W'].values[0], 'Height': df['H'].values[0], 'Resolution': 0.2}
        return building_detail_dict

    def _get_building_image(self):
        cv2_image = cv2.imread(self.building_image_path, cv2.IMREAD_GRAYSCALE)
        # flip the image
        image = cv2.flip(cv2_image, 0)
        # Look up table for transmission and reflection
        image[image == 50] = 1
        image[image == 200] = 2
        image[image == 254] = 3
        return image

    # implementation of matching simulated pathloss data to the building image
    # TODO: Implement this function of your own to get pathloss data, an example pathloss file is provided
    def get_pathloss_data(self, pathloss_file):
        resolution = float(pathloss_file.split('/')[-1].split('_')[-2])
        df = pd.read_csv(pathloss_file)
        df['X'] = (df['X'] - self.building_details['X_min']) / resolution
        df['Y'] = (df['Y'] - self.building_details['Y_min']) / resolution

        pathloss = np.zeros((int(self.building_details['Height']), int(self.building_details['Width'])))
        for i in range(len(df)):
            pathloss[round(df['Y'].values[i]), round(df['X'].values[i])] = df['PL'].values[i] if df['PL'].values[i] < 150 else 150

        return pathloss
