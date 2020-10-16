#pylint:skip-file
from torch.utils.data import Dataset
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True
class TripletDataset(Dataset):
    def __init__(self,MNIST):
        self.mnist=MNIST
        self.train=self.mnist.train
        self.transforms=self.mnist.transform
        if self.train:
            self.train_labels=self.mnist.train_labels
            self.train_data=self.mnist.train_data
            self.label_set=set(self.train_labels.numpy())
            self.label_to_indice={label: np.where(self.train_labels.numpy()==label) for label in self.label_set}
        else:
            self.test_label=self.mnist.test_labels
            self.test_data=self.mnist.test_data
            self.label_set=set(self.test_label.numpy())
            self.label_to_indice={label:np.where(self.test_data.numpy()==label) for label in self.label_set}
            random_state=np.random.RandomState(29)
            triplets=[[i,random_state.choice(self.test_label[i].item()),random_state.choice(self.label_to_indice[np.random.choice(list(self.label_set-set([self.test_label(i).item()])))])] for i in range(len(self.test_data))]

random_state=np.random.RandomState(29)
train_labels=torch.Tensor([1,2,3,4,5,6,7,8,9])
label_set=set(train_labels.numpy())
label_to_indices={label:np.where(train_labels.numpy()==label)[0]for label in label_set}
print(label_to_indices[train_labels[1].item()])
random_state.choice([1,2])
triplets = [[i,random_state.choice(label_to_indices[train_labels[i].item()]),random_state.choice(label_to_indices[np.random.choice(list(label_set - set([train_labels[i].item()])))])]for i in range(len(train_labels))]
print ('label_set:',label_set)
print ('label_to_indice:',label_to_indices)
print ('triplets:',triplets)