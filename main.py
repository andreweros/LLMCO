import os
import ast
import numpy as np
import sys
import anthropic
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from anthropic.types.message import Message
# This is where you could replace your own pathloss simulation function (Ray tracer)
from utils.pathloss_to_llm_coverter import calculate_coverage_on_pathloss_result
from utils.floor_plan_util import Building

# Set up the environment using anthropic API, can be replaced using OpenAI API
os.environ["ANTHROPIC_API_KEY"] = ""
env_folder = ""
# langchain.debug = True
np.set_printoptions(threshold=sys.maxsize)
client = anthropic.Anthropic()


def calculate_coverage_on_pathloss_result(files, building_name, threshold=100):

    # Create a building object
    building = Building(building_name)

    # Set up pathloss map at the same size of building and fill it with 150
    pathloss_map = np.full((int(building.building_details['Height']), int(building.building_details['Width'])), 150)
    coverage_map = np.zeros_like(pathloss_map)

    for file in files:
        # get pathloss data from each file
         pathloss_tmp = building.get_pathloss_data(file)
         # compare pathloss_tmp with pathloss, and keep the lowest pathloss
         pathloss_map = np.minimum(pathloss_map, pathloss_tmp)

    # Coverage map is a binary map, 1 for covered, 0 for not covered
    coverage_map[pathloss_map < threshold] = 1

    # Calculate the coverage percentage
    coverage = np.mean(coverage_map) * 100

    boundary_info = building.building_details

    # Reduce the boundary info by 20% (for visualization purpose)
    for key in boundary_info:
        boundary_info[key] = float(boundary_info[key]) * 0.9

    # Show pathloss map with color bar (0-150)
    plt.imshow(pathloss_map, cmap='jet', vmin=0, vmax=150)
    plt.title(f"Pathloss Map")
    plt.colorbar()
    plt.show()

    return coverage, boundary_info, pathloss_map, coverage_map

def calc_coverage(antenna_locations, building_name):

    # TODO: implement this function to calculate the coverage based on the given antenna locations
    coverage, boundary_info, pathloss_map, coverage_map = calculate_coverage_on_pathloss_result(building_name="", threshold=110)

    return coverage, boundary_info, pathloss_map, coverage_map


# function of fetching antenna locations from the result from LLM
def fetch_antenna_locations(building_info, data):
    location_index1 = data.rfind('[[')
    location_index2 = data.rfind(']]')
    if location_index1 == -1 or location_index2 == -1:
        return []

    data = data[location_index1:(location_index2 + 2)]

    locations = ast.literal_eval(data)

    # {'X_min': df['X_min'].values[0], 'X_max': df['X_max'].values[0],
    #                             'Y_min': df['Y_min'].values[0], 'Y_max': df['Y_max'].values[0],
    #                             'Width': df['W'].values[0], 'Height': df['H'].values[0], 'Resolution': 0.5}

    x_min = building_info["X_min"]
    y_min = building_info["Y_min"]
    resolution = building_info["Resolution"]
    width = building_info['Width']
    height = building_info['Height']

    z = 2
    locations_3d = [[location[0], location[1], z] for location in locations]

    return locations_3d, locations
# Chain of generating 1 AP location
def ap1_chain(inputs):
    message = client.messages.create(
        model=inputs["llm_model"],
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_prefix"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inputs["floorplan_image_type"],
                            "data": inputs["floorplan_image_data"],
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["ap1_request_desc"]
                    }
                ],
            }
        ],
    )

    return message

# Chain of generating N APs locations
def apn_chain(inputs):
    message: Message = client.messages.create(
        model=inputs["llm_model"],
        max_tokens=1024,
        # system="Respond only with integer number.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_prefix"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inputs["floorplan_image_type"],
                            "data": inputs["floorplan_image_data"],
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["ap_location_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_ratio_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_image_prefix"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inputs["coverage_image_type"],
                            "data": inputs["coverage_image_data"],
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_image_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["apn_request_desc"]
                    }
                ],
            }
        ],
    )

    return message

# Chain of optimization
def aco_chain(inputs):
    message = client.messages.create(
        model=inputs["llm_model"],
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_prefix"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inputs["floorplan_image_type"],
                            "data": inputs["floorplan_image_data"],
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["floorplan_image_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["ap_location_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_image_prefix"]
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inputs["coverage_image_type"],
                            "data": inputs["coverage_image_data"],
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_image_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["coverage_ratio_desc"]
                    },
                    {
                        "type": "text",
                        "text": inputs["aco_request_desc"]
                    }
                ],
            }
        ],
    )

    return message


def prepare_content(llm_model=None, floor_info=None, building_info=None, ap_locations=None, coverage_map=None,
                    coverage_ratio=None, ap_num=None, coverage_target=None):
    inputs = {}

    if llm_model is not None:
        inputs["llm_model"] = llm_model
    else:
        inputs["llm_model"] = "claude-3-sonnet-20240229"

    if floor_info is not None:
        floorplan_image = Image.fromarray(np.uint8(floor_info * 255), 'L')
        buffered = BytesIO()
        floorplan_image.save(buffered, format="JPEG")
        floor_width = building_info["X_max"] - building_info["X_min"]
        floor_depdth = building_info["Y_max"] - building_info["Y_min"]
        inputs["floorplan_image_type"] = "image/jpeg"
        inputs["floorplan_image_data"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
        inputs["floorplan_image_prefix"] = "You are given a floorplan image below: "
        inputs[
            "floorplan_image_desc"] = f'Here are some additional information of the floorplan image, ' \
                                      f'the size of the floorplan is {floor_width} m width and {floor_depdth} m depth. ' \
                                      f'The bottom left corner position is ({building_info["X_min"]}, {building_info["Y_min"]}). ' \
                                      f'The building contains wall partitions with doors and windows attached to it.'

        inputs[
            "ap1_request_desc"] = "Give me the location for one AP on the floor to maximise the wireless signal coverage. " \
                                  "The locations should be provided in meters and in the form of [[x1, y1], [x2, y2], ...] without any other text."

    if coverage_map is not None:
        coverage_image = Image.fromarray(np.uint8(coverage_map * 255), 'L')
        buffered = BytesIO()
        coverage_image.save(buffered, format="JPEG")
        inputs["coverage_image_prefix"] = "The detail coverage image is given below: "
        inputs["coverage_image_type"] = "image/jpeg"
        inputs["coverage_image_data"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
        inputs[
            "coverage_image_desc"] = "Here are some addtional inforamtion of the coverage image: " \
                                     "White color represents covered area and Black color represents area not covered."

    if coverage_ratio is not None:
        inputs["coverage_ratio_desc"] = f'You have achieved {coverage_ratio}% coverage'

    if ap_locations is not None:
        inputs["ap_location_desc"] = f'With the given AP locations below: {ap_locations}'

    if coverage_target is not None:
        inputs[
            "apn_request_desc"] = f'Based on the coverage achieved by 1 AP, please estimate how many APs are required to achieve ' \
                                  f'{coverage_target}% coverage. Provide your answer as an integer number only without any other text.'


    if ap_num is not None:
        inputs[
            "aco_request_desc"] = f'Give me new locations for {ap_num} APs on the floor that are differnent from all solutions above, ' \
                                  f'and has a higher coverage than any of the above. ' \
                                  f'Assume X is an integer and must be in the range of [{building_info["X_min"]},{building_info["X_max"]}], ' \
                                  f'Y is an integer and must be in the range of [{building_info["Y_min"]},{building_info["Y_max"]}]. ' \
                                  f'The locations should be provided in meters and in the form of [[x1, y1], [x2, y2], ...] without any other text.'

    return inputs


def lmco(floor_info, building_info, target_coverage, max_iterations=8):
    ap1_inputs = prepare_content(floor_info=floor_info, building_info=building_info)
    message = ap1_chain(ap1_inputs)

    # change string type lists '(2.5, 2.0, 2.5) \n (-5.5, -7.5, 2.5)' to list
    locations, locations_wh = fetch_antenna_locations(building_info, message.content[0].text)

    coverage, boundary_info, pathloss_map, coverage_map = calc_coverage(locations, building_name)

    histroy_df = pd.DataFrame({'location': [locations_wh], 'coverage': [coverage], 'coveragemap': [coverage_map]})

    apn_inputs = prepare_content(floor_info=floor_info, building_info=building_info,
                                 coverage_map=coverage_map,
                                 coverage_ratio=coverage, ap_locations=locations,
                                 coverage_target=target_coverage)

    message = apn_chain(apn_inputs)

    min_ap_num = int(message.content[0].text)
    max_ap_num = 20

    for location_num in range(min_ap_num, max_ap_num):

        for _ in range(max_iterations):

            aco_inputs = prepare_content(floor_info=floor_info, building_info=building_info,
                                         coverage_map=coverage_map,
                                         coverage_ratio=coverage, ap_locations=locations_wh,
                                         ap_num=location_num, coverage_target=target_coverage)

            message = aco_chain(aco_inputs)

            # change string type lists '[-15.5, 11.5, 3], [7.5, -9, 3]' to list
            locations, locations_wh = fetch_antenna_locations(building_info, message.content[0].text)
            if len(locations) != location_num:
                continue

            coverage, boundary_info, pathloss_map, coverage_map = calc_coverage(locations, building_name)

            # Concatenate the original DataFrame with the new DataFrame
            histroy_df.loc[len(histroy_df)] = [locations_wh, coverage, coverage_map]

            if coverage / target_coverage < 0.6:
                break

            if coverage >= target_coverage:
                print(f"Coverage reached {coverage}% with {location_num} locations at {locations}")
                return histroy_df
            else:
                print(f"Coverage reached {coverage}% with {location_num} locations at {locations}")

    return histroy_df


def get_2d_floor_info(building_name):
    floor = Building(building_name)
    floor_image_array = floor._get_building_image()
    return floor_image_array


if __name__ == '__main__':
    building_name = 'B1'
    target_coverage = 95

    building_info = Building(building_name)
    floor_info = get_2d_floor_info(building_name)
    performace_df = lmco(floor_info, building_info.building_details, target_coverage)

    coverage_array = performace_df["coverage"]
    # print(coverage_array)
    # Plot the progress based on the performance_df['coverage'] with steps
    plt.figure()
    plt.plot(coverage_array)
    plt.show()


