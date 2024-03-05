
name     = 'SIR_dnn_cdf_100k'
features = ['alpha', 'beta', 'lo']
target   = 'Zo'
nodes    =  16
gsize    =  4

import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(nodes, nodes), 
                                    nn.SiLU(),
                                    nn.Linear(nodes, nodes), 
                                    nn.SiLU())

    def forward(self, input):
        return self.module(input) + input


# Based on code written by ChatGPT v3.5
class GroupSort(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size
        
    def forward(self, input):
        """
        Splits input tensor into groups of size 'group_size' 
        and sorts each group independently.
    
        Args:
            input (torch.Tensor): The input tensor to be sorted.
    
        Returns:
            torch.Tensor: The sorted tensor, with elements 
            grouped and sorted in ascending order.
        """
        # Reshape the input tensor into groups of size 'group_size'
        grouped_tensor = input.view(-1, self.group_size)
    
        # Sort each group individually using torch.sort
        sorted_groups, _ = torch.sort(grouped_tensor)
        
        # Flatten the sorted tensor
        sorted_tensor = sorted_groups.reshape(input.shape)
    
        return sorted_tensor

model = nn.Sequential(nn.Linear(len(features), nodes), GroupSort(gsize),
                      nn.Linear(nodes, nodes), GroupSort(gsize),
                      nn.Linear(nodes, nodes), GroupSort(gsize),
                      nn.Linear(nodes, 1),  nn.Sigmoid()
                     )

# model = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(),
#                       nn.Linear(nodes, nodes), nn.ReLU(),
#                       nn.Linear(nodes, nodes), nn.SiLU(),
#                       nn.Linear(nodes, nodes), nn.ReLU(),
#                       nn.Linear(nodes, 1),  
#                       nn.Sigmoid()
#                      )
