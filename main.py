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
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
cuda=torch.cuda.is_available()

pars=argparse.ArgumentParser()
pars.add_argument('--train_mode',type=int,default=0,help='train modes')
par=pars.parse_args()

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)

class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


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
        self.total+=target[0].size(0)
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
        return self.nonlinear(self.embedding_net(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        return output1,output2
    
    def get_emdding(self,x):
        return self.embedding_net(x)



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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineTripletLoss(nn.Module):
    def __init__(self,margin,triplet_selector):
        super(OnlineTripletLoss,self).__init__()
        self.margin=margin
        self.triplet_selector=triplet_selector
    def forward(self,embeddings,target):
        triplets=self.triplet_selector.get_pairs(embeddings,target)
        if embeddings.is_cuda:
            triplets=triplets.cuda()
        ap_distances=(embeddings[triplets[:,0]]-embeddings[triplets[:,1]]).pow(2).sum(1)
        an_distances=(embeddings[triplets[:,0]]-embeddings[triplets[:,2]]).pow(2).sum(1)
        losses=F.relu(ap_distances-an_distances+self.margin)
        return losses.mean(), len(triplets)

def fit(train_loader,val_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,metrics=[],start_epoch=0):
    for epoch in range(0,start_epoch):
        scheduler.step()
    for epoch in range(start_epoch,n_epochs):
        scheduler.step()
        train_loss,metrics=train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics)
        message='Epoch {}/{}. Train set: Average loss:{:.4f}'.format(epoch+1,n_epochs,train_loss)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        val_loss,metrics=test_epoch(val_loader,model,loss_fn,cuda,metrics)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f}'.format(epoch+1,n_epochs,val_loss)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        print (message)

def train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics):
    for metric in metrics:
        metric.reset()
    
    model.train()
    losses=[]
    total_loss=0

    for batch_idx,(data,target) in enumerate(train_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        optimizer.zero_grad()
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        losses.append(loss.item())
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs,target,loss_outputs)
        
        if batch_idx%log_interval==0:
            message='Train:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(train_loader.dataset),100*batch_idx/len(train_loader),np.mean(losses))
            for metric in metrics:
                message+='\t{}:{}'.format(metric.name(),metric.value())
            
            print (message)
            losses=[]
    
    total_loss/=batch_idx+1
    return total_loss,metrics

def test_epoch(val_loader,model,loss_fn,cuda,metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
    model.eval()
    val_loss=0
    for batch_idx,(data,target) in enumerate(val_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        val_loss+=loss.item()

        for metric in metrics:
            metric(outputs,target,loss_outputs)
    
    return val_loss,metrics

mean,std=0.1307,0.3081
train_dataset=MNIST('../data/MNIST',train=True, download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]
))
test_dataset=MNIST('../data/MNIST',train=False,download=False,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]))
n_classes=10

mnist_classes=['0','1','2','3','4','5','6','7','8','9']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
def plot_embeddings(embeddings,targets,xlim=None,ylim=None):
    plt.figure(figsize=(10,10))
    for i in range (10):
        inds=np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0],embeddings[inds,1],alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(mnist_classes)
    plt.savefig(fig_path+'{:f}.png'.format(time.time()))
    plt.show()

def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            k+=len(images)
    return embeddings,labels


batch_size=256
kwargs={'num_workers':1,'pin_memory':True} if cuda else {}
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)

embedding_net=EmbeddingNet()
model=ClassificationNet(embedding_net,n_classes=n_classes)
if cuda:
    model=model.cuda()
loss_fn=nn.NLLLoss()
lr=1e-2
optimizer=optim.Adam(model.parameters(),lr=lr)
scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
n_epochs=20
log_interval=50

if par.train_mode==0:
    fit(train_loader,test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,metrics=[AccumulatedAccuracyMetric()])
    train_embeddings_baseline,train_labels_baseline=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_baseline,train_labels_baseline)
    val_embeddings_baseline,val_labels_baseline=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_baseline,val_labels_baseline)

if par.train_mode==1:
    siamese_train_dataset=SiameseMNIST(train_dataset)
    siamese_test_dataset=SiameseMNIST(test_dataset)
    batch_size=128
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    siamese_train_loader=DataLoader(siamese_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    siamese_test_loader=DataLoader(siamese_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=EmbeddingNet()
    model=SiameseNet(embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=20
    log_interval=100
    loss_fn=ContrastiveLoss(margin)

    fit(siamese_train_loader,siamese_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval)
    train_embeddings_sim,train_labels_sim=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_sim,train_labels_sim)
    val_embeddings_sim,val_labels_sim=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_sim,val_labels_sim)

if par.train_mode==2:
    triplet_train_dataset=TripletMNIST(train_dataset)
    triplet_test_dataset=TripletMNIST(test_dataset)
    batch_size=128
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    triplet_train_loader=DataLoader(triplet_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    triplet_test_loader=DataLoader(triplet_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=EmbeddingNet()
    model=TripletNet(embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=20
    log_interval=100
    loss_fn=TripletLoss(margin)

    fit(triplet_train_loader,triplet_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval)
    train_embeddings_triplet,train_labels_triplet=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_triplet,train_labels_triplet)
    val_embeddings_triplet,val_labels_triplet=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_triplet,val_labels_triplet)

    












        









