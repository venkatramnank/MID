import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)
        #TODO: Below the value of point_dim is changed from 2 to 3 to accomodate 3D points. Keep in Mind.
        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=3, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False), #TODO: Check dimension here
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'

            )
        )

    def encode(self, batch,node_type):
        
        #This below get_latent goes into trajectron
        z = self.encoder.get_latent(batch, node_type)
        #NOTE : This z is essentially f which encodes temporal and social information of size (256, 256)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        # import pdb;pdb.set_trace()
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type) # size  of input, batch size. This is f
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        feat_x_encoded = self.encode(batch,node_type) # B * 256
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
