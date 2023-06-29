
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d,ReLU,MaxPool2d,Linear
import dgl


class EdgeNetwork(nn.Module):

    def __init__(self,inputsize,hidden_layer_size,output_size,do_attention=True):
        s # what is missing here?
                
        self.do_attention = do_attention
        self.net = ### try 3 layers

    ### This function takes the set of all edges in the graph and prepares a message
    def forward(self, edges):
        
        # concatenate the
        # - features of the destination node
        # - hidden rep of the destination node
        # - features of the source node
        # - hidden rep of the source node
        # - the distance between the nodes (stored on the edge)
        input_data = torch.cat([ ... ],dim=1)

        out_dict = {} #this will be the returned message

        if self.do_attention:

            ### Self-attention between hidden rep of src and dst nodes
            k      = #key
            q      = #query
            sqrt_d = #square root of the dimension of the hidden representation
            attn   = #k-dot-q/sqrt_d
            attn   = torch.sigmoid(attn)
            edges.data['attn'] = attn #here we assign it to the edge data
            input_data = torch.cat([input_data, #it may help to add the attention to the input data (but on what dimension?)
            out_dict['attn']   = attn

        out_dict['node_h'] = #you want to pass the hidden rep of which node?
        out_dict['edge_h'] = #how will you compute the edge hidden rep? Hint: look in the __init__

        return out_dict

class NodeNetwork(nn.Module):

    def __init__(self,inputsize,hidden_layer_size,output_size,do_attention=True):
        super().__init__()

        self.do_attention = do_attention

        self.net = ### try 3 layers

    ### This function takes the set of all nodes in the graph and does operations on them using the messages from EdgeNetwork
    def forward(self, nodes):
        
        if self.do_attention:
            node_messages  = torch.sum(''' what goes here? '''*nodes.mailbox['attn'], dim=1)
        else:
            node_messages = torch.sum(''' same thing as here? ''',dim=1)
        
        edge_messages  = torch.sum(nodes.mailbox['edge_h'], dim=1)

        input_data = torch.cat([
            edge_messages,
            node_messages,
            nodes.data['node_h'],
            nodes.data['node_features']
            ],dim=1)
        
        node_hidden_rep = #Compute the new hidden rep of the node using the previous line and a neural network
        
        return {'node_h': node_hidden_rep }


class Classifier(nn.Module):

    def __init__(self,hid_rep_size=50,n_iterations=10,do_attention=True):
        super().__init__()
        
        self.n_iterations = n_iterations

        self.node_init = nn.Sequential(
            ### try 2 layers
        )

        ### updates edge hidden rep
        edge_input_features_size = 2*(hid_rep_size+2) + 1 + int(do_attention)
        self.edge_network = EdgeNetwork(inputsize=edge_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=hid_rep_size,do_attention=do_attention)

        ### updates node hidden rep
        node_input_features_size = 3*hid_rep_size + 2
        self.node_network = NodeNetwork(inputsize=node_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=hid_rep_size,do_attention=do_attention)

        ### predicts edge score
        # Here we can use the same handy EdgeNetwork class, we just need to set the output size to 1 (for the score)
        self.edge_classifier = EdgeNetwork(inputsize=edge_input_features_size,hidden_layer_size=2*hid_rep_size,output_size=1,do_attention=do_attention)
        
    def forward(self, g):
        
        g.ndata['node_h'] = #initialize the hidden rep of the nodes using one of the networks defined above
        g.edata['pred']   = torch.zeros(g.num_edges(),device=g.device)
        
        for i in range(self.n_iterations):
        
            # 1) Do the edge and node update sequence using update_all and the networks defined above
            ...
            # 2) Use apply_edges to classify each edge using our network
            ...
            # 3) Update the prediction by simply adding the score on each edge from (2) to the current prediction
            g.edata['pred'] += g.edata['edge_h'].view(-1)

        
        