#!/usr/bin/env python
import os
import re
import glob
import numpy as np
from matplotlib import pyplot as plt
from tabulate_yml_files import tabulate_data

folder_path = './calibs'
camera_folder_path = '*'
cali_path = 'calibrations'
videofolders = [f for f in glob.glob(f'{folder_path}/{camera_folder_path}/{cali_path}/*') if os.path.isdir(f)]

matrix_dict = {}

# Get all data from each dataset and save only the matrix parameters in a dictionary indicating camera type and video name
for vfolder in videofolders:
    data_coeffs = tabulate_data(vfolder)

    # Get only the camera matrix values
    matrix_params_idx = data_coeffs.index[:4].tolist()
    matrix_params = data_coeffs.to_numpy()[:4,:]
    
    dict_aux = {}
    for i in range(len(matrix_params_idx)):
        dict_aux[matrix_params_idx[i]] = matrix_params[i].tolist()
    
    # Assign a name to each video including the camera from where it was taken
    matrix_dict[f'{vfolder.split(os.sep)[1]}-{os.path.basename(vfolder)}'] = dict_aux

# Get the list with all available keys in matrix_dict dictionary
name_list = list(matrix_dict.keys())

# Get a list with only the cameras types and a dictionary with corresponding first and last indexes from the original list
camera_names_all = [item.split('-')[0] for item in name_list]
camera_names = list(dict.fromkeys(camera_names_all))
camera_names_pos = {t: (camera_names_all.index(t), len(camera_names_all) - 1 - camera_names_all[::-1].index(t)) for t in camera_names}

# Create a 2x2 subplot figure to include all 4 parameters next to each other
fig, ax = plt.subplots(2,2, figsize=(16, 9), layout='constrained')
ax = ax.flatten()

# Plot all parameters comparing them per camera and video
for idx in matrix_params_idx:
    i = matrix_params_idx.index(idx)

    # Add the average video value, and weight (1/sigma**2) value to a simplier list
    idx_average = [matrix_dict[item][idx][0] for item in name_list]
    idx_sigma = [matrix_dict[item][idx][1] for item in name_list]
    idx_weights = [1/(matrix_dict[item][idx][1])**2 for item in name_list]

    name_list_plot = []
    for camera in camera_names:
        # Get the indexes of the first and last element of the list related with the camera
        k = camera_names.index(camera)
        cam_st, cam_ed = camera_names_pos[camera]
        cam_st += k
        cam_ed += k

        # Get the lists of elements just for this specific camera
        camera_idx = [1 if camera in vid else 0 for vid in name_list]
        cam_average = np.array([vid for vid, mask in zip(idx_average, camera_idx) if mask])
        cam_weights = np.array([vid for vid, mask in zip(idx_weights, camera_idx) if mask])

        # Plot the value (average) for each video and add an error bar (sigma)
        for j in range(len(name_list)):
            if camera_idx[j]:
                ax[i].scatter(idx_average[j], j+k)
                ax[i].errorbar(idx_average[j], j+k, xerr=idx_sigma[j])
                name_list_plot.append(name_list[j])

        # Calculate the weighted average and standard deviation
        average_cam_idx = np.average(cam_average, weights=cam_weights)
        std_cam_idx = np.sqrt(np.sum(1/cam_weights)/len(cam_weights))

        # Add average and std results for the camera to the plot
        ax[i].scatter(average_cam_idx, cam_ed+1, color='black')
        ax[i].errorbar(average_cam_idx, cam_ed+1, xerr=std_cam_idx, color='black')
        name_list_plot.append(f'Average {camera}')

        # Plot the vertical line in the average position (over X-axis) and put the camera and value on top or bottom of the line
        av_text = cam_st-1.5 if (camera_names.index(camera) % 2) else cam_ed+1.5 # -1.5 to fit the text under the first point, +1.5 to fit the text over the average point which is not count in cam_ed
        ax[i].vlines(average_cam_idx, cam_st, cam_ed+1, linestyles=(0, (1, 1)), colors='black')
        ax[i].text(average_cam_idx, av_text, f'{camera}: {average_cam_idx:.2f}', color='black', ha='center', va='bottom', fontsize=9)
        
        print(idx, camera, f'{average_cam_idx:.2f}', f'{std_cam_idx:.4f}')

    # Set names to every element in y-axis
    ax[i].set_yticks(np.arange(len(name_list_plot)))
    ax[i].set_yticklabels(name_list_plot)
    ax[i].set_xlabel('pixel')

    # Set title
    ax[i].set_title(f'Parameter: {idx}')
plt.savefig(f'{folder_path}/result_params.png', dpi=1200)
plt.show()