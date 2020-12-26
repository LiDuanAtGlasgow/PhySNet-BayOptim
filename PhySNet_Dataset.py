#pylint:skip-file
from torch.vision import VisionDataset
from PIL import Image
import os
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union

class PhySNet_Dataset(VisionDataset):
    training_file=''
    test_file=''
    classes=[]
    def __init__(self,root:str,train:bool=True,trainsform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(PhySNet_Dataset,self).__init__(root,transform=trainsform,target_transform=target_transform)
        self.train=train
        if self.train:
            data_file=self.train_file
        else:
            data_file=self.test_file
        self.data,self.targets=torch.load(os.path.join(self.processed_folder,data_file))
        if not self._check_exists():
            raise RuntimeError('You have to load your data!')
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        img,target=self.data[index],int(self.targets[index])
        img-Image.fromarray(img.numpy(),mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
    return img, target

    def __len__(self)->int:
        return len(self.data)
    @property
    def class_to_idx(self)->Dict[str,str]:
        return {_class: i for i, _class in enumerate(self.classes)}
    def _check_exists(self)->bool:
        return os.path.exists(os.path.join(self.processed_folder,self.train_file)) and os.path.exists(os.path.join(self.processed_folder,self.test_file))

    

