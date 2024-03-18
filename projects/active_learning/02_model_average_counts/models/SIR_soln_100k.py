
name     = 'SIR_soln_100k'
features = ['alpha', 'beta', 't']
target   = ['s', 'i']
nodes    =  20

import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(len(features), nodes), nn.ReLU(),
                      nn.Linear(nodes, nodes), nn.ReLU(),
                      nn.Linear(nodes, nodes), nn.ReLU(),
                      nn.Linear(nodes, nodes), nn.ReLU(),
                      nn.Linear(nodes, 2))
