def get_traj_hypers():
    hypers = {   'batch_size': 256,
    'grad_clip': 1.0,
    'learning_rate_style': 'exp',
    'min_learning_rate': 1e-05,
    'learning_decay_rate': 0.9999,
    'prediction_horizon': 12,
    'minimum_history_length': 1,
    'maximum_history_length': 7,
    'map_encoder':
        {'PEDESTRIAN':
            {'heading_state_index': 6,
             'patch_size': [50, 10, 50, 90],
             'map_channels': 3,
             'hidden_channels': [10, 20, 10, 1],
             'output_size': 32,
             'masks': [5, 5, 5, 5],
             'strides': [1, 1, 1, 1],
             'dropout': 0.5
            }
        },
    'k': 1,
    'k_eval': 25,
    'kl_min': 0.07,
    'kl_weight': 100.0,
    'kl_weight_start': 0,
    'kl_decay_rate': 0.99995,
    'kl_crossover': 400,
    'kl_sigmoid_divisor': 4,
    'rnn_kwargs':
        {'dropout_keep_prob': 0.75},
    'MLP_dropout_keep_prob': 0.9,
    'enc_rnn_dim_edge': 128,
    'enc_rnn_dim_edge_influence': 128,
    'enc_rnn_dim_history': 128,
    'enc_rnn_dim_future': 128,
    'dec_rnn_dim': 128,
    'q_z_xy_MLP_dims': None,
    'p_z_x_MLP_dims': 32,
    'GMM_components': 1,
    'log_p_yt_xz_max': 6,
    'N': 1,
    'tau_init': 2.0,
    'tau_final': 0.05,
    'tau_decay_rate': 0.997,
    'use_z_logit_clipping': True,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 5.0,
    'z_logit_clip_crossover': 300,
    'z_logit_clip_divisor': 5,
    'dynamic':
        {'PEDESTRIAN':
            {'name': 'SingleIntegrator',
             'distribution': False,
             'limits': {}
            }
        },
    'state':
        {'PEDESTRIAN':
            {'position': ['x', 'y'],
             'velocity': ['x', 'y'],
             'acceleration': ['x', 'y']
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
    'log_histograms': False,
    'dynamic_edges': 'yes',
    'edge_state_combine_method': 'sum',
    'edge_influence_combine_method': 'attention',
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'offline_scene_graph': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'edge_encoding': True,
    'use_map_encoding': False,
    'augment': True,
    'override_attention_radius': [],
    'learning_rate': 0.01,
    'npl_rate': 0.8,
    'K': 80,
    'tao': 0.4}
    return hypers



def get_traj_hypers_3D():
    hypers = {   'batch_size': 256,
    'grad_clip': 1.0,
    'learning_rate_style': 'exp',
    'min_learning_rate': 1e-05,
    'learning_decay_rate': 0.9999,
    'prediction_horizon': 12,
    'minimum_history_length': 1,
    'maximum_history_length': 7,
    'map_encoder':
        {'PEDESTRIAN':
            {'heading_state_index': 6,
             'patch_size': [50, 10, 50, 90], # TODO: need to confirm with Dr Chanho
             'map_channels': 3,
             'hidden_channels': [10, 20, 10, 1],
             'output_size': 32,
             'masks': [5, 5, 5, 5],
             'strides': [1, 1, 1, 1],
             'dropout': 0.5
            }
        },
    'k': 1,
    'k_eval': 25,
    'kl_min': 0.07,
    'kl_weight': 100.0,
    'kl_weight_start': 0,
    'kl_decay_rate': 0.99995,
    'kl_crossover': 400,
    'kl_sigmoid_divisor': 4,
    'rnn_kwargs':
        {'dropout_keep_prob': 0.75},
    'MLP_dropout_keep_prob': 0.9,
    'enc_rnn_dim_edge': 128,
    'enc_rnn_dim_edge_influence': 128,
    'enc_rnn_dim_history': 128,
    'enc_rnn_dim_future': 128,
    'dec_rnn_dim': 128,
    'q_z_xy_MLP_dims': None,
    'p_z_x_MLP_dims': 32,
    'GMM_components': 1,
    'log_p_yt_xz_max': 6,
    'N': 1,
    'tau_init': 2.0,
    'tau_final': 0.05,
    'tau_decay_rate': 0.997,
    'use_z_logit_clipping': True,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 5.0,
    'z_logit_clip_crossover': 300,
    'z_logit_clip_divisor': 5,
    'dynamic':
        {'PEDESTRIAN':
            {'name': 'SingleIntegrator',
             'distribution': False,
             'limits': {}
            }
        },
    'state':
        {'PEDESTRIAN':
            {'position': ['x', 'y', 'z'],
             'velocity': ['x', 'y', 'z'],
             'acceleration': ['x', 'y', 'z']
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y', 'z']}},
    'log_histograms': False,
    'dynamic_edges': 'yes',
    'edge_state_combine_method': 'sum',
    'edge_influence_combine_method': 'attention',
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'offline_scene_graph': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'edge_encoding': True,
    'use_map_encoding': False,
    'augment': True,
    'override_attention_radius': [],
    'learning_rate': 0.01,
    'npl_rate': 0.8,
    'K': 80,
    'tao': 0.4}
    return hypers