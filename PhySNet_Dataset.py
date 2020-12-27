#pylint:skip-file
from torch.vision import VisionDataset
from PIL import Image
import os
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import pandas as pd

class PhySNet_Dataset(VisionDataset):
    classes=[]
    def __init__(self,root:str,train:bool=True,trainsform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(PhySNet_Dataset,self).__init__(root,transform=trainsform,target_transform=target_transform)
        self.train=train
        if self.train:
            data_file=self.train_file
        else:
            data_file=self.test_file
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        img,target=self.data[index],int(self.targets[index])
        img=Image.fromarray(img.numpy(),mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
    return img, target

    

