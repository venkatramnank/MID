import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle
import math

from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.033 #30 fps

standardization = {
    'PEDESTRIAN': {
        'top_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'top_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'top_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
        'bottom_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'bottom_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'bottom_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
        'front_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'front_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'front_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
        'back_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'back_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'back_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
        'right_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'right_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'right_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
        'left_position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean': 0, 'std': 1}
        },
        'left_velocity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z' : {'mean' : 0, 'std': 1}
        },
        'left_acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            'z': {'mean': 0, 'std': 1}
        },
 
    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    """Scene augmentor using angles
    """
    def rotate_pc(pc, alpha):
        
         
        #TODO: Need to change the angle only across the y axis (gravity)
        #TODO: Also need to augment translation on x-z plane
        R = np.array([
            [np.cos(alpha * np.pi / 180), 0, -np.sin(alpha * np.pi / 180)],
            [0, 1, 0],
            [np.sin(alpha * np.pi / 180), 0, np.cos(alpha * np.pi / 180)]
        ])
        # import pdb; pdb.set_trace()
        # T = np.array([np.random.uniform(-1.0, 1.0), 0, np.random.uniform(-1.0, 1.0)])
        # RT = np.eye(4)
        # RT[:3, :3] = R
        # MatT = np.eye(4)
        # MatT[:3, 3] = T
        #TODO: need to do translation later
        return R @ pc

    data_columns = pd.MultiIndex.from_product([['top_position','bottom_position', 'front_position', 'back_position',
                                            'right_position','left_position','top_velocity','bottom_velocity',
                                            'front_velocity','back_velocity','right_velocity','left_velocity',
                                            'top_acceleration',
                                              'bottom_acceleration',
                                              'front_acceleration',
                                              'back_acceleration',
                                              'left_acceleration',
                                            'right_acceleration'], ['x', 'y', 'z']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle *np.pi / 180
    
    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()
        z = node.data.position.z.copy()
        

        x, y, z = rotate_pc(np.array([x, y, z]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        vz = derivative_of(z, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)
        az = derivative_of(vz, scene.dt)
        

        data_dict = {('position', 'x'): x,
                    ('position', 'y'): y,
                    ('position', 'z'): z,
                    ('velocity', 'x'): vx,
                    ('velocity', 'y'): vy,
                    ('velocity', 'z'): vz,
                    ('acceleration', 'x'): ax,
                    ('acceleration', 'y'): ay,
                    ('acceleration', 'z'): az}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

data_folder_name = 'processed_data'

maybe_makedirs(data_folder_name) # build data folder 

#multiple levels of data with position, vel, acc as level 1. Each of level will have x and y values
"""MultiIndex([(    'position', 'x'),
        (    'position', 'y'),
        (    'velocity', 'x'),
        (    'velocity', 'y'),
        ('acceleration', 'x'),
        ('acceleration', 'y')],
        )
"""
data_columns = pd.MultiIndex.from_product([['top_position','bottom_position', 'front_position', 'back_position',
                                            'right_position','left_position','center_position','top_velocity','bottom_velocity',
                                            'front_velocity','back_velocity','right_velocity','left_velocity', 'center_velocity',
                                            'top_acceleration',
                                              'bottom_acceleration',
                                              'front_acceleration',
                                              'back_acceleration',
                                              'left_acceleration',
                                            'right_acceleration',
                                              'center_acceleration'], ['x', 'y', 'z']])
#TODO: See the environment representation

# Process ETH-UCY
for desired_source in ['collide']:
    for data_class, rot_t_data_class in zip(['train', 'val', 'test'], ['collide_train_rot_trans_train', 'collide_train_rot_trans_val', 'collide_test_rot_trans']):
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        """ 
        {'scenes': None, 'node_type_list': ['PEDESTRIAN'], 'attention_radius': {(PEDESTRIAN, PEDESTRIAN): 3.0}, 
        'NodeType': [PEDESTRIAN], 'robot_type': None, 'standardization': {'PEDESTRIAN': {'position': {'x': {'mean': 0, 'std': 1},
        'y': {'mean': 0, 'std': 1}}, 'velocity': {'x': {'mean': 0, 'std': 2}, 
        'y': {'mean': 0, 'std': 2}}, 'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}}}, 
        'standardize_param_memo': {}, '_scenes_resample_prop': None}
        """
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

        subdir = '/home/kalyanav/MS_thesis/MID/raw_data/collide/'
        subdir_rt = '/home/kalyanav/MS_thesis/MID/raw_data/collide/'                          
        files = sorted(next(os.walk(os.path.join('raw_data', desired_source, data_class)))[2])
        files_rt = sorted(next(os.walk(os.path.join('raw_data', desired_source, rot_t_data_class)))[2])

        # for (subdir, dirs, files), (subdir_rt, dirs_rt, files_rt) in zip(os.walk(os.path.join('raw_data', desired_source, data_class)), os.walk(os.path.join('raw_data', desired_source, rot_t_data_class))):
        for file, file_rt in zip(files, files_rt):
            
            input_data_dict = dict()
            full_data_path = os.path.join(subdir + data_class, file)
            full_data_path_rt = os.path.join(subdir + rot_t_data_class, file_rt)
            print('At', full_data_path)
           
            data = pd.read_csv(full_data_path, sep=' ', index_col=False, header=None)
            data.columns = ['frame_id', 'track_id', 'top_x', 'top_y', 'top_z',
                                    'bottom_x', 'bottom_y', 'bottom_z',
                                    'front_x', 'front_y', 'front_z',
                                    'back_x', 'back_y', 'back_z',
                                    'left_x', 'left_y', 'left_z',
                                    'right_x', 'right_y', 'right_z'] #in each txt file
            data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
            data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

            data['frame_id'] = data['frame_id'] 

            data['frame_id'] -= data['frame_id'].min()

            data['node_type'] = 'PEDESTRIAN'
            data['node_id'] = data['track_id'].astype(str)

            # at this point you have the following columns:
            # frame_id, track_id, pos_x, pos_y, node_type, node_id
            data.sort_values('frame_id', inplace=True)

            if desired_source == "eth" and data_class == "test":
                data['pos_x'] = data['pos_x'] * 0.6
                data['pos_y'] = data['pos_y'] * 0.6

            # if data_class == "train":
            #     #data_gauss = data.copy(deep=True)
            #     data['pos_x'] = data['pos_x'] + 2 * np.random.normal(0,1)
            #     data['pos_y'] = data['pos_y'] + 2 * np.random.normal(0,1)

                #data = pd.concat([data, data_gauss])

            # mean standardization
            
            # data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
            # data['pos_y'] = data['pos_y'] - data['pos_y'].mean()
            # data['pos_z'] = data['pos_z'] - data['pos_z'].mean()
            

            max_timesteps = data['frame_id'].max() # frame id based maximum timesteps
            
            # Reading the R and t matrix
            R_t_objects = []
            with open(full_data_path_rt, "rb") as openfile:
                while True:
                    try:
                        R_t_objects.append(pickle.load(openfile))
                    except EOFError:
                        break
            reshaped_array_R_t = np.reshape(R_t_objects[0], (len(pd.unique(data['frame_id'])), len(pd.unique(data['node_id'])), 4, 4))
            
            
            scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=None)
            """
            scene class contains the following:
            {'map': None, 'timesteps': 594, 'dt': 0.4, 'name': 'eth_train', 'nodes': [], 'robot': None, 'temporal_scene_graph': None, 
            'frequency_multiplier': 1, 'description': '', 'aug_func': <function augment at 0x7f71d6931a80>, 
            'non_aug_scene': None}
            """
            
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]

                node_values = node_df[['top_x', 'top_y', 'top_z',
                                    'bottom_x', 'bottom_y', 'bottom_z',
                                    'front_x', 'front_y', 'front_z',
                                    'back_x', 'back_y', 'back_z',
                                    'left_x', 'left_y', 'left_z',
                                    'right_x', 'right_y', 'right_z']].values

                if node_values.shape[0] < 3:
                    continue
                
                current_r_t_data = reshaped_array_R_t[:, int(node_id), :, :]
                
                new_first_idx = node_df['frame_id'].iloc[0]
                
                top_x = node_values[:, 0] # x.shape : (20,)
                top_y = node_values[:, 1] # y.shape : (20,)
                top_z = node_values[:, 2]
                
                bottom_x = node_values[:, 3] # x.shape : (20,)
                bottom_y = node_values[:, 4] # y.shape : (20,)
                bottom_z = node_values[:, 5]
                
                front_x = node_values[:, 6] # x.shape : (20,)
                front_y = node_values[:, 7] # y.shape : (20,)
                front_z = node_values[:, 8]
                
                back_x = node_values[:, 9] # x.shape : (20,)
                back_y = node_values[:, 10] # y.shape : (20,)
                back_z = node_values[:, 11]
                
                left_x = node_values[:, 12] # x.shape : (20,)
                left_y = node_values[:, 13] # y.shape : (20,)
                left_z = node_values[:, 14]
                
                right_x = node_values[:, 15] # x.shape : (20,)
                right_y = node_values[:, 16] # y.shape : (20,)
                right_z = node_values[:, 17]


                # Assuming you have already defined node_values and scene.dt
                # Calculate the derivatives of positions
                top_vx = np.gradient(top_x, scene.dt)
                top_vy = np.gradient(top_y, scene.dt)
                top_vz = np.gradient(top_z, scene.dt)

                bottom_vx = np.gradient(bottom_x, scene.dt)
                bottom_vy = np.gradient(bottom_y, scene.dt)
                bottom_vz = np.gradient(bottom_z, scene.dt)

                front_vx = np.gradient(front_x, scene.dt)
                front_vy = np.gradient(front_y, scene.dt)
                front_vz = np.gradient(front_z, scene.dt)

                back_vx = np.gradient(back_x, scene.dt)
                back_vy = np.gradient(back_y, scene.dt)
                back_vz = np.gradient(back_z, scene.dt)

                left_vx = np.gradient(left_x, scene.dt)
                left_vy = np.gradient(left_y, scene.dt)
                left_vz = np.gradient(left_z, scene.dt)

                right_vx = np.gradient(right_x, scene.dt)
                right_vy = np.gradient(right_y, scene.dt)
                right_vz = np.gradient(right_z, scene.dt)
                

                # Calculate the derivatives of velocities to get accelerations
                top_ax = np.gradient(top_vx, scene.dt)
                top_ay = np.gradient(top_vy, scene.dt)
                top_az = np.gradient(top_vz, scene.dt)

                bottom_ax = np.gradient(bottom_vx, scene.dt)
                bottom_ay = np.gradient(bottom_vy, scene.dt)
                bottom_az = np.gradient(bottom_vz, scene.dt)

                front_ax = np.gradient(front_vx, scene.dt)
                front_ay = np.gradient(front_vy, scene.dt)
                front_az = np.gradient(front_vz, scene.dt)

                back_ax = np.gradient(back_vx, scene.dt)
                back_ay = np.gradient(back_vy, scene.dt)
                back_az = np.gradient(back_vz, scene.dt)

                left_ax = np.gradient(left_vx, scene.dt)
                left_ay = np.gradient(left_vy, scene.dt)
                left_az = np.gradient(left_vz, scene.dt)

                right_ax = np.gradient(right_vx, scene.dt)
                right_ay = np.gradient(right_vy, scene.dt)
                right_az = np.gradient(right_vz, scene.dt)

                # Update the data_dict with the new values
                data_dict= {
                    ('top_position', 'x'): top_x,
                    ('top_position', 'y'): top_y,
                    ('top_position', 'z'): top_z,
                    ('bottom_position', 'x'): bottom_x,
                    ('bottom_position', 'y'): bottom_y,
                    ('bottom_position', 'z'): bottom_z,
                    ('front_position', 'x'): front_x,
                    ('front_position', 'y'): front_y,
                    ('front_position', 'z'): front_z,
                    ('back_position', 'x'): back_x,
                    ('back_position', 'y'): back_y,
                    ('back_position', 'z'): back_z,
                    ('left_position', 'x'): left_x,
                    ('left_position', 'y'): left_y,
                    ('left_position', 'z'): left_z,
                    ('right_position', 'x'): right_x,
                    ('right_position', 'y'): right_y,
                    ('right_position', 'z'): right_z,
                    ('top_velocity', 'x'): top_vx,
                    ('top_velocity', 'y'): top_vy,
                    ('top_velocity', 'z'): top_vz,
                    ('bottom_velocity', 'x'): bottom_vx,
                    ('bottom_velocity', 'y'): bottom_vy,
                    ('bottom_velocity', 'z'): bottom_vz,
                    ('front_velocity', 'x'): front_vx,
                    ('front_velocity', 'y'): front_vy,
                    ('front_velocity', 'z'): front_vz,
                    ('back_velocity', 'x'): back_vx,
                    ('back_velocity', 'y'): back_vy,
                    ('back_velocity', 'z'): back_vz,
                    ('left_velocity', 'x'): left_vx,
                    ('left_velocity', 'y'): left_vy,
                    ('left_velocity', 'z'): left_vz,
                    ('right_velocity', 'x'): right_vx,
                    ('right_velocity', 'y'): right_vy,
                    ('right_velocity', 'z'): right_vz,
                    ('top_acceleration', 'x'): top_ax,
                    ('top_acceleration', 'y'): top_ay,
                    ('top_acceleration', 'z'): top_az,
                    ('bottom_acceleration', 'x'): bottom_ax,
                    ('bottom_acceleration', 'y'): bottom_ay,
                    ('bottom_acceleration', 'z'): bottom_az,
                    ('front_acceleration', 'x'): front_ax,
                    ('front_acceleration', 'y'): front_ay,
                    ('front_acceleration', 'z'): front_az,
                    ('back_acceleration', 'x'): back_ax,
                    ('back_acceleration', 'y'): back_ay,
                    ('back_acceleration', 'z'): back_az,
                    ('left_acceleration', 'x'): left_ax,
                    ('left_acceleration', 'y'): left_ay,
                    ('left_acceleration', 'z'): left_az,
                    ('right_acceleration', 'x'): right_ax,
                    ('right_acceleration', 'y'): right_ay,
                    ('right_acceleration', 'z'): right_az
                }
                # import pdb; pdb.set_trace()
                node_data = pd.DataFrame(data_dict, columns=data_columns)
                node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data, r_t_data= reshaped_array_R_t[:, int(node_id), :, :])
                """
                node object contains:
                {'type': PEDESTRIAN, 'id': '1', 'length': None, 'width': None, 'height': None, 'first_timestep': 0, 'non_aug_node': None, 
                'data': <environment.data_structures.DoubleHeaderNumpyArray object at 0x7f717173fb90>, 
                'is_robot': False, '_last_timestep': None, 'description': '', 'frequency_multiplier': 1, 'forward_in_time_on_next_override': False}
                
                """
                node.first_timestep = new_first_idx

                scene.nodes.append(node)
                
                # if data_class == 'train':
                #     scene.augmented = list()
                #     angles = np.arange(0, 360, 90) if data_class == 'train' else [0]
                    
                #     for angle in angles:
                        
                #         scene.augmented.append(augment_scene(scene, angle))

                print(scene)
                scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
exit()
# Process Stanford Drone. Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


for data_class in ["train", "test"]:
    raw_path = "raw_data/stanford"
    out_path = "processed_data"
    data_path = os.path.join(raw_path, f"{data_class}_trajnet.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(out_path, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")

    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        data['x'] = data['x'] - data['x'].mean()
        data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:

            scene = Scene(timesteps=max_timesteps+1, dt=dt, name="sdd_" + data_class, aug_func=augment if data_class == 'train' else None)
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        pdb.set_trace()

                    node_values = node_df[['x', 'y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                 ('position', 'y'): y,
                                 ('velocity', 'x'): vx,
                                 ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): ax,
                                 ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            if data_class == 'train':
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            #pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
