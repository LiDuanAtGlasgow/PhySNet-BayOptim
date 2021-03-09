#type:ignore
from matplotlib.pyplot import figure
from pandas.core.frame import DataFrame
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement,qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
import numpy as np
import pandas as pd
import csv
import time
from matplotlib import pyplot as plt

np.random.seed(42)

bias=[-0.25091976,0.90142861,0.46398788,0.19731697,-0.68796272,-0.68801096,-0.88383278,0.73235229,0.20223002,0.41614516,
-0.95883101,0.9398197,0.66488528,-0.57532178,-0.63635007,-0.63319098,-0.39151551]
bias=torch.Tensor(bias)
bias=bias.unsqueeze(dim=0)

def Bayesian_Search(data=None):
    d=17
    bounds = torch.stack([-1*torch.ones(d), torch.ones(d)])
    parameters=torch.from_numpy(data)
    physical_distance=-1*(parameters-bias).pow(2).sum(1)
    physical_distance=physical_distance.unsqueeze(dim=1)

    gp = SingleTaskGP(parameters,physical_distance)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    sampler = SobolQMCNormalSampler(1000)
    #print ('physical_distance:',physical_distance.shape)
    best_f=0
    #qUCB = qUpperConfidenceBound(gp, 0.1, sampler)
    #UCB = UpperConfidenceBound(gp, beta=0.1)
    qEI = qExpectedImprovement(gp, best_f, sampler)
    candidate, acq_value = optimize_acqf(qEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,)

    return candidate

def read_parameters(csv_file='./bayoptim.csv'):
    data=pd.read_csv(csv_file)
    (width,length)=data.shape
    paramters=np.zeros((width,length-1))
    for i in range(len(paramters)):
        for t in range (len(paramters[i])):
            paramters[i][t]=data.iloc[i,t+1]
    return paramters

def get_parameters(parameters):
    data=pd.read_csv('./bayoptim.csv')
    index=data.shape[0]
    parameters=parameters.squeeze(dim=0).cpu().numpy()
    with open ('./bayoptim.csv','a') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow((str(index),parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],
        parameters[11],parameters[12],parameters[13],parameters[14],parameters[15],parameters[16]))

def datas(source_path):
    data=pd.read_csv(source_path)
    bending_stiffness_mean=np.zeros((data.shape[0],))
    bending_stiffness_std=np.zeros((data.shape[0],))
    for i in range(len(bending_stiffness_mean)):
        bending_stiffness_mean[i]=np.mean(data.iloc[i,1:18])
        bending_stiffness_std[i]=np.std(data.iloc[i,1:18])
    return bending_stiffness_mean.tolist(),bending_stiffness_std.tolist()

def figure_plot(data,name,color):
    x_axis=np.zeros((len(data),))
    for i in range(len(data)):
        x_axis[i]=i
    
    df=pd.DataFrame({'x':x_axis,'data':data})
    figure=plt.figure()
    subplt=figure.add_subplot(111)
    subplt.plot('x','data',data=df,color=color,label=name)
    subplt.set_xlabel('Epoch')
    subplt.set_ylabel(name)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title(name+'_plot')
    plt.savefig('./'+name+str(time.time())+'.png')

start_time=time.time()
for i in range (200):
    data=read_parameters()
    candidate=Bayesian_Search(data=data)
    get_parameters(candidate)
    if i%10==0:
        print (f'{i}/200 has been finished, time:{time.time()-start_time}')
        start_time=time.time()

mean,std=datas('./bayoptim.csv')
figure_plot(mean,'mean','red')
figure_plot(std,'std','blue')
plt.show()

print ('finished!')
