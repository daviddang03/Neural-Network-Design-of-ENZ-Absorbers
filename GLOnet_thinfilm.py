import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from net import ResGeneratorFixed
from datetime import datetime
from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from utils import Params
import vectorized_tmm_dispersive_multistack as tmm   
#------------------------------------------------------------------------------------------------

def Tensor_TMM_Solve(wavelength,n_list,thickness,theta,pol='p'):
    Spectral_data=tmm.coh_vec_tmm_disp_mstack(pol, n_list, thickness, theta, wavelength,device='cpu')
    T_list=Spectral_data['T']
    R_list=Spectral_data['R']
    return T_list,R_list


class GLOnet():
    def __init__(self, params):     
        self.dtype = torch.FloatTensor
        
        # construct
        if params.net =='Res_Fixed':
            self.generator= ResGeneratorFixed(params)
        
 
        #self.generator.cuda()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr, betas = (params.beta1, params.beta2), weight_decay = params.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = params.step_size, gamma = params.gamma)
        
        # training parameters
        self.noise_dim = params.noise_dim
        self.numIter = params.numIter
        self.batch_size = params.batch_size
        self.sigma = params.sigma
        self.alpha_sup = params.alpha_sup
        self.iter0 = 0
        self.alpha = 0.1
    
        # simulation parameters
        self.n_database = params.n_database
        self.k = params.k.type(self.dtype)  # number of frequencies
        self.theta = params.theta.type(self.dtype) # number of angles
        self.pol = params.pol # str of pol
        self.target_absorption = params.target_absorption.type(self.dtype)
        self.s=params.s
        self.b1=params.b1
        self.b2=params.b2
        
        # tranining history
        self.loss_training = []
        self.refractive_indices_training = []
        self.thicknesses_training = []
        
        
    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.
        
    def sample_z(self, batch_size):
        return (torch.randn(batch_size, self.noise_dim, requires_grad=True)).type(self.dtype)
    
    def global_loss_function(self,reflection,transmission):
        loss_1=-2*torch.mean(torch.exp(-torch.mean(torch.pow((1-reflection-transmission)[:,:,self.b1:self.b2] - (self.target_absorption)[self.b1:self.b2], 2),dim=(1))))
        loss_2=-torch.mean(torch.exp(-torch.mean(torch.pow((-reflection-transmission)[:,:,:self.b1], 2),dim=(1))))
        loss_3=-torch.mean(torch.exp(-torch.mean(torch.pow((-reflection-transmission)[:,:,self.b2:], 2),dim=(1))))
        return loss_1+loss_2+loss_3
    
    def record_history(self, loss, thicknesses, refractive_indices):
        self.loss_training.append(loss.detach())
        self.thicknesses_training.append(thicknesses.mean().detach())
        self.refractive_indices_training.append(refractive_indices.mean().detach())
        
    def viz_training(self):
        plt.figure(figsize = (20, 5))
        plt.subplot(131)
        plt.plot(self.loss_training)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
    
        
    def train_fixed(self):
      
        self.generator.train()   # training loop
        
        with tqdm(total=self.numIter) as t:
            it = self.iter0  
            while True:
                it +=1 
                # normalized iteration number
                normIter = it / self.numIter

                # discretizaton coeff.
                self.update_alpha(normIter) 

                # terminate the loop
                if it > self.numIter:
                    return 

                # sample z
                z = self.sample_z(self.batch_size)
    
                # generate a batch of optical parameters
                thicknesses, drude_indices,carrier_density = self.generator(z, self.alpha)
                reflection=torch.zeros((thicknesses.size()[0],self.theta.size()[0],self.k.size()[0]))
                transmission=torch.zeros((thicknesses.size()[0],self.theta.size()[0],self.k.size()[0]))

                # calculate efficiencies and gradients using EM solver
                for i in range(thicknesses.size()[0]): 
                    trans, reflect=Tensor_TMM_Solve(2*math.pi/self.k,drude_indices[i],thicknesses[i],self.theta,self.pol)
                    transmission[i,:]=trans
                    reflection[i,:]=reflect
                
                self.optimizer.zero_grad()

                # construct the loss 
                g_loss = self.global_loss_function(reflection,transmission)

                # record history
                self.record_history(g_loss, thicknesses, drude_indices)
                
                # train the generator
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # update progress bar
                t.update()               


    def evaluate_fixed(self, num_devices):
        self.generator.eval()
        z = self.sample_z(num_devices)
        thicknesses, drude_indices,carrier_density= self.generator(z, self.alpha)
        trans, reflect=Tensor_TMM_Solve(2*math.pi/self.k,drude_indices,thicknesses,self.theta)
        thicknesses=thicknesses.detach().numpy()
        drude_indices=drude_indices.detach().numpy()
        transmission=trans.detach().numpy()
        reflection=reflect.detach().numpy()
        carrier_density=carrier_density.detach().numpy()
        return thicknesses,drude_indices,transmission, reflection,carrier_density