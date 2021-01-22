#type:ignore
import os 
import json
import csv
import pandas as pd
import numpy as np
def get_parameters(parameters):
    data=pd.read_csv('./parameters.csv')
    index=data.shape[0]
    with open ('./parameters.csv','a') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow((str(index),parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],
        parameters[11],parameters[12],parameters[13],parameters[14],parameters[15],parameters[16]))
    print ('get_parameters finished!')
            

class Get_ArcSim_Script():
    def __init__(self,bend_stiffness,winds,density,index):
        self.bend_stiffness=bend_stiffness
        self.winds=winds
        self.density=density
        self.index=index
        self.material_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/materials/materials_bayoptim_%d.json'%self.index
        self.conf_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/conf/conf_bayoptim_%d.json'%self.index
    
    def make_materials(self):
        data={}
        data['density']=self.density
        data['stretching']=[]
        data['stretching'].append([2002.08618, 0.345588, 2027.48559, 19.189022])
        data['stretching'].append([2107.22753, 36.371597, 2171.73339, 16.904713])
        data['stretching'].append([2107.29882, 57.936157, 2153.30712, 24.392872])
        data['stretching'].append([2107.73291, 28.389059, 2132.72607, 101.462021])
        data['stretching'].append([2106.93750, 70.149529, 2121.87353, 38.054455])
        data['stretching'].append([2107.30786, 61.216290, 2150.47680, 37.668823])
        data['bending']=np.zeros((3,5))
        #print ('data:',data['bending'][0])
        for i in range (len(data['bending'])):
            for j in range (len(data['bending'][i])):
                data['bending'][i][j]=self.bend_stiffness[i][j]
        data['bending']=data['bending'].tolist()
        with open (self.material_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def make_conf(self):
        data={}
        data['frame_time']=0.04
        data["frame_steps"]=8
        data["duration"]=20
        data["cloths"]=[]
        data['cloths'].append({
        "mesh": "meshes/square.obj",
        "transform": {
        "translate": [0, 0, 0],
        "rotate": [120, 1, 1, 1]},
        "materials": [{
        "data": 'materials/materials_bayoptim_'+str(self.index)+'.json',
        "thicken": 1,
        "strain_limits": [0.95, 1.05]
        }],
        "remeshing": {
        "refine_angle": 0.3,
        "refine_compression": 0.01,
        "refine_velocity": 1,
        "size": [20e-3, 500e-3],
        "aspect_min": 0.2
        }
        })
        data["motions"]=[]
        data["handles"]=[{"nodes": [2,3]}]
        data["gravity"]=[0, 0, -9.8]
        data["wind"]={"velocity": [self.winds,self.winds, 0]}
        data['magic']={"repulsion_thickness": 10e-3, "collision_stiffness": 1e6}
        with open (self.conf_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def forward(self):
        self.make_materials()
        self.make_conf()
        print ('make_materials and make_conf are completed')

def read_parameters(csv_file='./parameters.csv'):
    data=pd.read_csv(csv_file)
    (width,length)=data.shape
    print ('data:',data.shape)
    paramters=np.zeros((width,length-1))
    for i in range(len(paramters)):
        for t in range (len(paramters[i])):
            paramters[i][t]=data.iloc[i,t+1]
    return paramters



