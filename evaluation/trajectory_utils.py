import numpy as np
import pdb
from utils.history_future_pred_utils import *

def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
           
            position_state = {'top_position': ['x', 'y', 'z'],
             'bottom_position': ['x', 'y', 'z'],
             'front_position': ['x', 'y', 'z'],
             'back_position': ['x', 'y', 'z'],
             'left_position': ['x', 'y', 'z'],
             'right_position': ['x', 'y', 'z']
            } #TODO: Need to change!! All center calculations and rotation to be applied here
            # 
            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]
            #pdb.set_trace()
            future = node.get(np.array([t + 1, t + ph]), position_state)
            # replace nan to 0
            #future[np.isnan(future)] = 0
            future = future[~np.isnan(future.sum(axis=1))]

            # if prune_ph_to_future:
            #     # predictions_output = predictions_output[:, :, :future.shape[0]]
            #     prediction_output_new = recursively_apply_rotation_translation(history, predictions_output)
            #     if predictions_output.shape[2] == 0:
            #         continue
            prediction_output_new = recursively_apply_rotation_translation(history, predictions_output)
            trajectory = prediction_output_new
            history_updated = convert_six_to_mean_arr(history)
            future_updated = convert_six_to_mean_arr(future)

            if map is None:
                histories_dict[t][node] = history_updated
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future_updated
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict
