#pylint:skip-file
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data.sampler import BatchSampler
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations

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
            self.triplets=triplets

    def __getitem__(self,index):
        if self.train:
            img1,label1=self.train_data[index],self.train_labels[index].item()
            positive_index=index
            while positive_index==index:
                positive_index=np.random.choice(self.label_to_indice[label1])
            negative_label=np.random.choice(list(self.label_set-set([label1])))
            negative_index=np.random.choice(self.label_to_indice[negative_label])
            img2=self.train_data[positive_index]
            img3=self.train_labels[negative_index]
        else:
            img1=self.test_data[self.triplets[index][0]]
            img2=self.test_data[self.triplets[index][1]]
            img3=self.test_data[self.triplets[index][2]]
        
        img1=Image.fromarray(img1.numpy(),mode='L')
        img2=Image.fromarray(img2.numpy(),mode='L')
        img3=Image.fromarray(img3.numpy(),mode='L')

        if self.transforms is not None:
            img1=self.transforms(img1)
            img2=self.transforms(img2)
            img3=self.transforms(img3)
        
        return (img1,img2,img3),[]

class BalancedBatchSampler(BatchSampler):
    def __init__(self,labels,n_classes,n_samples):
        self.labels=labels
        self.label_set=list(set(self.labels.numpy()))
        self.label_to_indice={label:np.where(self.labels.numpy()==label) for label in self.label_set}
        for i in self.label_set:
            np.random.shuffle(self.label_to_indice[i])
        self.used_label_indice_count={label:0 for labels in self.label_set}
        self.count=0
        self.n_classes=n_classes
        self.n_samples=n_samples
        self.n_dateset=len(self.label_set)
        self.batch_size=self.n_classes*self.n_samples
    
    def __iter__(self):
        self.count=0
        while self.count+self.batch_size<self.n_dateset:
            classes=np.random.choice(self.label_set,self.n_classes,replace=False)
            indices=[]
            for class_ in classes:
                indices.extend(self.label_to_indice[class_][self.used_label_indice_count[class_]:self.used_label_indice_count[class_]+self.n_samples])
                self.used_label_indice_count[class_]+=self.n_samples
                if self.used_label_indice_count[class_]+self.n_samples>self.label_to_indice[class_]:
                    np.random.shuffle(self.label_to_indice[class_])
                    self.used_label_indice_count[class_]=0
            yield indices
            self.count+=self.n_classes*self.n_samples
    
    def __len__(self):
        return self.n_dateset//self.batch_size

class TripletLoss(nn.Module):
    def __init__(self,margin):
        super(TripletLoss,self).__init__()
        self.margin=margin
    
    def forward (self,anchor,positive,negative,size_average=True):
        distance_positive=(anchor-positive).pow(2).sum()
        distance_negative=(anchor-negative).pow(2).sum()
        losses=F.relu(distance_positive-distance_negative+self.margin)
        return losses.mean() if size_average else losses.sum()

class Metric:
    def __init__(self):
        pass

    def __call__(self,outputs,target,loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class AccumulatedAccuracyMetric(Metric):
    def __init__(self):
        self.correct=0
        self.total=0
    
    def __call__(self,outputs,target,loss):
        pred=outputs[0].data.max(1,keepdim=True)[1]
        self.correct+=pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total+=targetp[0].size(0)
        return self.value()
    
    def reset(self):
        self.correct=0
        self.total=0
    
    def value(self):
        return 100*(self.correct/self.total)
    
    def name(self):
        return 'Accuracy'

class AverageNonzeroTripletMetric(Metric):
    def __init__(self):
        self.values=[]
    
    def __call__(self,outputs,target,loss):
        self.values.append(loss[1])
        return self.value()
    
    def reset(self):
        self.values=[]
    
    def value(self):
        return np.mean(self.values)
    
    def name(self):
        return 'Average Non-Zero Triplets'

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet,self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,5),
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(64*4*4,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.convnet(x)
        output=output.reshape(output.size()[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2,self).__init__()
    
    def forward(self,x):
        output=super(EmbeddingNetL2,self).forward(x)
        output=output.pow(2).sum(1,keepdim=True).sqrt()
        return output

    def get_emdding(self,x):
        return self.forward(x)
    
class ClassificationNet(nn.Module):
    def __init__(self,embedding_net,n_classes):
        super(ClassificationNet,self).__init__()
        self.embedding_net=embedding_net
        self.n_classes=n_classes
        self.nonlinear=nn.PReLU()
        self.fc1=nn.Linear(2,n_classes)
    
    def forward(self,x):
        output=self.embedding_net(x)
        output=self.nonlinear(output)
        scores=F.log_softmax(self.fc1(output),dim=-1)
        return scores
    
    def get_emdding(self,x):
        return self.nonlinear(self.embedding_net)

class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)

def pdist(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix

class PairSelector:
    def __init__(self):
        pass

    def get_pairs(self,embeddings,labels):
        raise NotImplementedError

class AllPositivePairSelector(PairSelector):
    def __init__(self,balance=True):
        super(AllPositivePairSelector,self).__init__()
        self.balance=balance
    
    def get_pairs(self,embeddings,labels):
        labels=labels.cpu().numpy()
        all_pairs=np.array(list(combinations(range(len(labels),2))))
        all_pairs=torch.LongTensor(all_pairs)
        positive_pairs=all_pairs[(labels[all_pairs[:,0]==all_pairs[:,1]]).nonzero()]
        negative_pairs=all_pairs[(labels[all_pairs[:,0]!=all_pairs[:,1]]).nonzero()]
        if self.balance:
            negative_pairs=negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
        
        return positive_pairs, negative_pairs

class HardNegativeSelector(PairSelector):
    def __init__(self,cpu=True):
        super(HardNegativeSelector,self).__init__()
        self.cpu=cpu
    
    def get_pairs(self,embeddings,labels):
        if self.cpu:
            embeddings=embeddings.cpu()
        distance_matrix=pdist(embeddings)
        labels=labels.cpu.numpy()
        all_pairs=np.array(list(combinations(range(len(labels),2))))
        all_pairs=torch.LongTensor(all_pairs)
        positive_pairs=all_pairs[(labels[all_pairs[:,0]==all_pairs[:,1]]).nonzero()]
        negative_pairs=all_pairs[(labels[all_pairs[:,0]!=all_pairs[:,1]]).nonzero()]
        negative_distances=distance_matrix[negative_pairs[:,0],negative_pairs[:,1]]
        negative_distances=negative_distances.cpu().numpy()
        top_negatives=np.argpartition(negative_distances,len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs=negative_pairs[torch.LongTensor(top_negatives)]
        return positive_pairs, top_negative_pairs
    
class TripletSelector:
    def __init__(self):
        pass

    def get_triplets(self,embeddings,labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):
    def __init__(self):
        super(AllTripletSelector,self).__init__()
    
    def get_triplets(self,embeddings,labels):
        labels=labels.cpu().numpy()
        triplets=[]
        for label in set(labels):
            label_mask=(labels==labels)
            label_indices=np.where(label_mask)[0]
            if len(label_indices)<2:
                continue
            negative_indices=np.where(np.logical_not(label_mask))[0]
            anchor_positives=list(combinations(label_indices,2))
            temp_triplets=[[anchor_positive[0],anchor_positive[1],neg_in] for anchor_positive in anchor_positives for neg_in in negative_indices]
            triplets+=temp_triplets
        return torch.LongTensor(np.array(triplets))

    def hardest_negative(loss_values):
        hard_negative=np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative]>0 else None

    def random_hard_negative(loss_values):
        rd_negative=np.where(loss_values>0)[0]
        return np.random.choice(rd_negative) if len(rd_negative)>0 else None

class FunctionNegativeTripletSelector(TripletSelector):
    def __init__(self,margin,negative_selection_fn,cpu=True):
        super(FunctionTripletSelector,self).__init__()
        self.cpu=cpu
        self.margin=margin
        self.negative_selection_fn=negative_selection_fn
        
    def get_triplets(self,embeddings,labels):
        if self.cpu:
            embeddings=embeddings.cpu()
        distance_matrix=pdist(embeddings)
        distance_matrix=distance_matrix.cpu()

        labels=labels.cpu().numpy()
        triplets=[]

        for label in set(labels):
            label_mask=(labels==label)
            label_indices=np.where(label_mask)[0]
            if len(label_indices)<2:
                continue
            negative_indices=np.where(np.logical_not(label_mask))[0]
            anchor_positives=list(combinations(label_indices,2))
            anchor_positives=np.array(anchor_positives)

            ap_distances=distance_matrix[anchor_positives[:,0],anchor_positives[:,1]]
            for anchor_positive,ap_distance in zip(anchor_positives,ap_distances):
                loss_values=ap_distance-distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])),torch.LongTensor(negative_indices)]+self.margin
                loss_values=loss_values.cpu().numpy()
                hard_negative=self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative=negative_indices[hard_negative]
                    triplets.append(anchor_positive[0],anchor_positive[1],hard_negative)
                    
        if len(triplets)==0:
            triplets.append(anchor_positive[0],anchor_positivep[1],negative_indices[0])
            
        triplets=np.array(triplets)
        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=hardest_negative,cpu=cpu)

def RandomNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=cpu)

def SemihardNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=lambda x:semihard_negative(x,margin),cpu=cpu)

'''
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
labels=torch.Tensor([0,1,2,3,4,5,6,7,8,9])
label_set=set(labels.numpy())
label_to_indice={label:np.where(labels.numpy()==label) for label in label_set}
labels=torch.Tensor([1,2,3,1,2,3,1,2,3,1,2,3]).to(device)
labels=labels.cpu().numpy()
for label in set(labels):
    label_mask=(labels==label)
    label_indices=np.where(label_mask)[0]
    if len(label_indices)<2:
        continue
    negative_indices=np.where(np.logical_not(label_mask))[0]
    anchor_positives=list(combinations(label_indices,2))
    anchor_positives=np.array(anchor_positives)
    anchor_0=anchor_positives[:,0]
    anchor_1=anchor_positives[:,1]

vectors=torch.rand(3,2)
print ('vector:',vectors.shape)
pdist_value=pdist(vectors)
distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
vectors_t=torch.t(vectors)
vectors_m=vectors.mm(torch.t(vectors))
vectors_p=vectors.pow(2).sum(dim=1).view(-1,1)+vectors.pow(2).sum(dim=1).view(1,-1)
difference_sqaure=(vectors.view(1,-1)-vectors.view(-1,1))*(vectors.view(1,-1)-vectors.view(-1,1))
print ('vector_t:',vectors_t.shape)
print ('vector_m:',vectors_m.shape)
print ('vector_p:',vectors_p.shape)
print ('distance_matrix:',distance_matrix)
print ('difference_square',difference_sqaure)

x=torch.Tensor([[1,2],[3,4]])
print('x[1,2]',x[0,1])
'''




        









