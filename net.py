import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as math
from utils import Params
from material_database import MatDatabase
import torch


def interp_wv_complex(wv_in_t,wv_ex_t,refractive):
    '''
        parameters
            wv_in (tensor) : number of wavelengths
            material_key (list) : number of materials

        return
            refractive indices (tensor or tuple of tensor) : number of materials x number of wavelengths
    ''' 
    wv_in=torch.clone(wv_in_t).detach().numpy()
    wv_ex=wv_ex_t
    refractive_real=torch.clone(refractive.real).detach().numpy()
    refractive_imag=torch.clone(refractive.imag).detach().numpy()
        
    n_data = np.zeros(np.shape(wv_in)[0])
    k_data = np.zeros(np.shape(wv_in)[0])
    n_data[:] = np.interp(wv_in, wv_ex, refractive_real)
    k_data[:] = np.interp(wv_in, wv_ex, refractive_imag)
    return torch.tensor(n_data+1j*k_data,dtype=torch.complex128)

def Drude(carrier_density,wavelength,mobility=14*10**-4):
    #Takes in a tensor for the carrier density
    #Wavelength should be in microns
    pi=math.pi
    lambd=wavelength*10**-6 #convert microns to meters
    frequency=3*10**8/lambd
    omega=2*pi*frequency
    epsilon_inf=3.9
    epsilon_0=8.85*10**-12 
    m_star=0.28*9.10938188*10**-31 #kg
    e=1.60217646*10**-19 #kg


    Gamma=e/m_star/mobility
    lambda_gamma=(1*10**9)*2*math.pi*(3*10**8)/Gamma
    n = carrier_density*10**6
    omega_p =torch.sqrt(n*e**2/(epsilon_0*m_star))*(10**10)
    lambda_p = (1*10**9)*2*math.pi*(3*10**8)/omega_p


    omega_ENZ=torch.sqrt(omega_p**2/epsilon_inf-Gamma**2)
    lambda_ENZ = (1*10**9)*2*pi*(3*10**8)/omega_ENZ
    omega_ENZ0=torch.sqrt(omega_p**2/epsilon_inf)
    lambda_ENZ0 = (1*10**9*2)*pi*(3*10**8)/omega_ENZ0
    omega=omega.to(dtype=torch.cdouble)
    epsilon = epsilon_inf-(omega_p**2)/(omega**2+1j*omega*Gamma)


    epsilon_real = torch.real(epsilon)
    epsilon_imag = torch.imag(epsilon)

    index_n = torch.sqrt(epsilon_real/2+0.5*torch.sqrt(epsilon_real**2+epsilon_imag**2))
    index_k = torch.sqrt(-epsilon_real/2+0.5*torch.sqrt(epsilon_real**2+epsilon_imag**2))
    nk=index_n+1j*index_k
    nk=nk.to(dtype=torch.cdouble)
    return nk



class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=16):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(dim, dim*2, bias=False),
                nn.BatchNorm1d(dim*2),
                nn.LeakyReLU(0.2),
                nn.Linear(dim*2, dim, bias=False),
                nn.BatchNorm1d(dim))

    def forward(self, x):
        return F.leaky_relu(self.block(x) + x, 0.2)





class ResGeneratorFixed(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.noise_dim = params.noise_dim
        self.res_layers = params.res_layers
        self.res_dim = params.res_dim
        self.N_layers = params.N_layers
        self.M_materials = params.M_materials
        self.n_database = params.n_database.view(1, 1, params.M_materials, -1) # 1 x 1 x number of mat x number of freq
        self.k=params.k
        self.s=params.s

        self.carrier_int=params.carrier_int
        self.carrier_fin=params.carrier_fin
        self.carrier_step=params.carrier_step

        self.thickness_int=params.thickness_int
        self.thickness_fin=params.thickness_fin
        self.thickness_step=params.thickness_step
        
        
        self.num_of_carrier=torch.arange(self.carrier_int,self.carrier_fin,self.carrier_step).size()[0]
        self.carrier_range=torch.arange(self.carrier_int,self.carrier_fin,self.carrier_step)

        self.num_of_thickness=torch.arange(self.thickness_int,self.thickness_fin,self.thickness_step).size()[0]
        self.thickness_range=torch.arange(self.thickness_int,self.thickness_fin,self.thickness_step)

        self.mobility=params.mobility
        
        
        self.initBLOCK = nn.Sequential(
            nn.Linear(self.noise_dim, self.res_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2)
        )

        self.endBLOCK = nn.Sequential(
            nn.Linear(self.res_dim, self.N_layers*(self.num_of_carrier+self.num_of_thickness), bias=False),
            nn.BatchNorm1d(self.N_layers*(self.num_of_carrier+self.num_of_thickness)),
        )        
                
        self.ResBLOCK = nn.ModuleList()
        for i in range(params.res_layers):
            self.ResBLOCK.append(ResBlock(self.res_dim))
        
        self.FC_thickness = nn.Sequential(   

            nn.Linear(self.N_layers, 36,bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(36,64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.N_layers,bias=False),
            nn.Tanh()
        )
        

    def forward(self, noise, alpha):
        waves=2*math.pi/self.k

        #Generate a large tensor for the probability values for the carrier density and thicknesses
        net = self.initBLOCK(noise)
        for i in range(self.res_layers):
            self.ResBLOCK[i](net)
        carrier_thickness_tensor = self.endBLOCK(net)
        carrier_thickness_tensor_reshaped=carrier_thickness_tensor.view(-1,self.N_layers,self.num_of_carrier+self.num_of_thickness)



        #Splits the previous tensor into separate carrier density probabilities and the thicknesses probabilities
        refractive_matrix=carrier_thickness_tensor_reshaped[:,:,0:self.num_of_carrier]
        thickness_matrix=carrier_thickness_tensor_reshaped[:,:,self.num_of_carrier:]

        #Apply softmax activation function
        Prob_Carrier_Density=F.softmax(refractive_matrix*alpha,dim=2)
        Prob_Thickness=F.softmax(thickness_matrix*alpha,dim=2)

        #Gives the final weighted and summed values of the carrier densities and thicknesses
        weighted_carrier_sum = torch.sum(Prob_Carrier_Density*self.carrier_range,dim=2)
        weighted_thickness_sum=torch.sum(Prob_Thickness*self.thickness_range,dim=2)

        #Generate empty tensors for preallocation
        drude_indices=torch.zeros((weighted_thickness_sum.size()[0],weighted_thickness_sum.size()[1],self.n_database.size()[-1]),dtype=torch.complex128)
        carrier_density=torch.zeros((weighted_thickness_sum.size()[0],weighted_thickness_sum.size()[1]-2))
        drude_indices[:,0,:]=self.n_database[0,0,0,:]
        drude_indices[:,-1,:]=self.n_database[0,0,-1,:]


        thicknesses=torch.zeros((weighted_carrier_sum.size()[0],weighted_thickness_sum.size()[1]))
        thicknesses[:,0]=np.inf
        thicknesses[:,-1]=np.inf

        #-------------------------------
        carrier_value=weighted_carrier_sum
        thicknesses[:,1:-1]=weighted_thickness_sum[:,1:-1]
        #-------------------------------
        net_in=carrier_value

        for idx in range(net.size()[0]):
            for i in range(1,self.N_layers-1):  
                    refractive=Drude(net_in[idx,i],waves,self.mobility) 
                    drude_indices[idx,i,:]=refractive
            carrier_density[idx,:]=net_in[idx,1:-1]
        return (thicknesses, drude_indices,carrier_density)
