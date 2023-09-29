# from utils import prediction_output_to_trajectories
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from evaluation.trajectory_utils import prediction_output_to_trajectories


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'm--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)
    
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting functionality
import numpy as np

def plot_3d_trajectories(ax,
                          prediction_dict,
                          histories_dict,
                          futures_dict,
                          line_alpha=0.7,
                          line_width=0.2,
                          edge_width=2,
                          node_sphere_radius=0.3,
                          batch_num=0,
                          kde=False):
    
    cmap = ['k', 'b', 'y', 'g', 'r']
    #TODO(venkat) : fix the history issue
    
    for node in histories_dict:
        
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue
        
        
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    # Replace this with a 3D KDE plot if needed
                    ax.scatter(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1], predictions[batch_num, :, t, 2],
                                c=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1], predictions[batch_num, sample_num, :, 2],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    future[:, 2],
                    'm--')

            # Current Node Position
            ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2],
                        c='g', s=node_sphere_radius * 100, edgecolors='k', linewidth=1, alpha=1.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('auto')


def plot_3d_trajectories_sep_plots(
                          prediction_dict,
                          histories_dict,
                          futures_dict,
                          line_alpha=0.7,
                          line_width=0.2,
                          edge_width=2,
                          node_sphere_radius=0.3,
                          batch_num=0,
                          kde=False):
    
    cmap = ['k', 'b', 'y', 'g', 'r']
    fig = plt.figure(figsize=(12, 8))  # Create a single figure for all subplots
    
    # Create a grid of subplots to display trajectories
    num_nodes = len(histories_dict)
    num_cols = 3  # Number of columns for subplots (you can adjust this)
    num_rows = (num_nodes + num_cols - 1) // num_cols
    
    for i, node in enumerate(histories_dict):
        
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue
        
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')  # Create a new subplot
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    # Replace this with a 3D KDE plot if needed
                    ax.scatter(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1], predictions[batch_num, :, t, 2],
                                c=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1], predictions[batch_num, sample_num, :, 2],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    future[:, 2],
                    'm--')

            # Current Node Position
            ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2],
                        c='g', s=node_sphere_radius * 100, edgecolors='k', linewidth=1, alpha=1.0)

    # Add labels and adjust the layout
    fig.text(0.5, 0.04, 'X', ha='center')
    fig.text(0.04, 0.5, 'Y', va='center', rotation='vertical')
    fig.suptitle('3D Trajectories')
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

def visualize_3d_prediction(i, scene_path, prediction_output_dict,
                            dt,
                            max_hl,
                            ph,
                            robot_node=None,
                            map=None,
                            **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    #TODO: Fix and understand the output properly
    
    # import pdb; pdb.set_trace()
    ts_key = list(prediction_dict.keys())[0]
        
    prediction_dict_curr = prediction_dict[ts_key]
    histories_dict_curr = histories_dict[ts_key]
    futures_dict_curr = futures_dict[ts_key]

      # Create a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    ax.view_init(azim=-90, elev=135)

    # if map is not None:
    #     # You may need to adjust this based on your map data
    #     ax.scatter(map[:, 0], map[:, 1], map[:, 2], c='gray', marker='.')

    plot_3d_trajectories(ax, prediction_dict_curr, histories_dict_curr, futures_dict_curr, *kwargs)
    plt.tight_layout()
    plt.savefig(scene_path + str(i) + '.png')
    # plt.show()
    plt.close()
