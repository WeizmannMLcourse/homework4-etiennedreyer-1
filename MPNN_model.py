
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d,ReLU,MaxPool2d,Linear
import dgl


class EdgeNetwork(nn.Module):

    def __init__(self,inputsize,hidden_layer_size,output_size,do_attention=True):
        super().__init__()
                
        self.do_attention = do_attention
        self.net = nn.Sequential(
            nn.Linear(inputsize,hidden_layer_size),nn.ReLU(),
            nn.Linear(hidden_layer_size,hidden_layer_size),nn.ReLU(),
            nn.Linear(hidden_layer_size,output_size)
            )

    def forward(self, edges):
        
        input_data = torch.cat([edges.dst['node_features'],
                                edges.dst['node_h'],
                                edges.src['node_features'],
                                edges.src['node_h'],
                                edges.data['distance'].unsqueeze(1)]
                                ,dim=1)

        out_dict = {}

        if self.do_attention:

            ### Self-attention between hidden rep of src and dst nodes
            k      = edges.dst['node_h']
            q      = edges.src['node_h']
            sqrt_d = torch.sqrt(torch.tensor(k.shape[1],dtype=torch.float32))
            attn   = torch.sum(k*q,dim=1,keepdim=True)/sqrt_d
            attn   = torch.sigmoid(attn)
            edges.data['attn'] = attn
            input_data = torch.cat([input_data,attn],dim=1)
            out_dict['attn']   = attn

        out_dict['node_h'] = edges.src['node_h']
        out_dict['edge_h'] = self.net(input_data)

        return out_dict

    
class NodeNetwork(nn.Module):

    def __init__(self,inputsize,hidden_layer_size,output_size,do_attention=True):
        super().__init__()

        self.do_attention = do_attention

        self.net = nn.Sequential(
            nn.Linear(inputsize,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,output_size)
            )

    def forward(self, nodes):
        
        if self.do_attention:
            node_messages  = torch.sum(nodes.mailbox['node_h']*nodes.mailbox['attn'], dim=1)
        else:
            node_messages = torch.sum(nodes.mailbox['node_h'],dim=1)
        
        edge_messages  = torch.sum(nodes.mailbox['edge_h'], dim=1)

        input_data = torch.cat([
            edge_messages,
            node_messages,
            nodes.data['node_h'],
            nodes.data['node_features']
            ],dim=1)
        
        node_hidden_rep = self.net(input_data)
        
        return {'node_h': node_hidden_rep }


class Classifier(nn.Module):

    def __init__(self,hid_rep_size=50,do_attention=True):
        super().__init__()
        
        self.node_init = nn.Sequential(
            nn.Linear(2,hid_rep_size//2),
            nn.ReLU(),
            nn.Linear(hid_rep_size//2,hid_rep_size)
        )

        ### updates edge hidden rep
        edge_input_features_size = 2*(hid_rep_size+2) + 1 + int(do_attention)
        self.edge_network = EdgeNetwork(inputsize=edge_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=hid_rep_size,do_attention=do_attention)

        ### updates node hidden rep
        node_input_features_size = 3*hid_rep_size + 2
        self.node_network = NodeNetwork(inputsize=node_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=hid_rep_size,do_attention=do_attention)

        ### predicts edge score
        self.edge_classifier = EdgeNetwork(inputsize=edge_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=1,do_attention=do_attention)
        
    def forward(self, g):
        
        g.ndata['node_h'] = self.node_init(g.ndata['node_features'])
        #g.edata['attn']   = torch.zeros(g.num_edges(),device=g.device)
        g.edata['pred']   = torch.zeros(g.num_edges(),device=g.device)
        
        for i in range(10):
            g.update_all(self.edge_network,self.node_network)
            g.apply_edges(self.edge_classifier)

            g.edata['pred'] += g.edata['edge_h'].view(-1)

        
        