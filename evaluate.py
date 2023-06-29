import torch
import torch.nn as nn
import numpy as np
from dataset import ShortestPathDataset, collate_graphs
from MPNN_model import Classifier
import sys
from torch.utils.data import Dataset, DataLoader

def compute_f1_and_loss(dataloader,net):
    
    edge_true_pos = 0
    edge_false_pos = 0
    edge_false_neg = 0

    pos_weight = torch.Tensor([10])
    if torch.cuda.is_available():
        pos_weight = pos_weight.to(torch.device('cuda'))

    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loss = 0
    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    n_batches = 0
    with torch.no_grad():
        for batched_g in dataloader:
            n_batches+=1
            
            if torch.cuda.is_available():
                batched_g = batched_g.to(torch.device('cuda'))
                
            net(batched_g)
            
            edge_target = batched_g.edata['on_path']
            edge_pred = batched_g.edata['pred']

            loss+= loss_func(edge_pred,edge_target).item()
            edge_pred = torch.sigmoid(edge_pred)
            
            thresh = 0.5
            edge_true_pos +=len(torch.where( (edge_pred>thresh) & (edge_target==1) )[0])
            edge_false_pos+=len(torch.where( (edge_pred>thresh) & (edge_target==0) )[0])
            edge_false_neg+=len(torch.where( (edge_pred<thresh) & (edge_target==1) )[0])
            
    f1 = edge_true_pos/(edge_true_pos+0.5*(edge_false_pos+edge_false_neg)+1e-6)
    loss = loss/n_batches      
    return f1, loss


def evaluate_on_dataset(path_to_ds=None):


	test_ds = ShortestPathDataset(path_to_ds)
	dataloader = DataLoader(test_ds,batch_size=300,collate_fn=collate_graphs)


	net = Classifier()
	net.load_state_dict(torch.load('trained_model.pt',map_location=torch.device('cpu')))

	f1, loss = compute_f1_and_loss(dataloader,net)
	
	return f1, loss


if __name__ == "__main__":

	f1, loss = evaluate_on_dataset(sys.argv[1])

	print(f1)

