
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Naming convention:
B is the batch size
N is the number of objects, typically the number of other traffic participants
D is the feature dimensionality, i.e. the number of channels in feature vectors or in feature maps
"""


class LSTMForGNN(nn.Module):
    def __init__(self, Din, D):
        self.layer = nn.Linear(Din + D, 4 * D)
        self.D = D
    def _mask(self, output, updated_c, mask):
        mask = mask.view(B, N, 1).expand(-1, -1, self.D)
        output = output.where(mask, torch.zeros_like(output))
        updated_c = output.where(mask, torch.zeros_like(updated_c))        
    def forward(self, vertices, mask, vertex_state):
        B, N, Din = vertices.size()
        if vertex_state is None:
            vertex_state = [torch.zeros((B, N, self.D), device=vertices.device) for i in range(2)]
        z = torch.cat([
            vertices,
            vertex_state[0]
        ], dim=2)
        z = self.layer(z)
        i = torch.sigmoid(z[:, :, 0  : D])
        f = torch.sigmoid(z[:, :, 2*D: 3*D])
        o = torch.sigmoid(z[:, :, D  : 2*D])
        c = F.tanh(z[:, :, 3*D: 4*D])
        updated_c = i * c + f * vertex_state[1]
        output = o * F.tanh(updated_c)
        output, updated_c = self._mask(output, updated_c, mask)
        return output, [output, updated_c]

class GraphBlock(nn.Module):
    def __init__(self, message_layer, node_layer):
        super().__init__()
        self.message_layer = message_layer
        self.node_layer = node_layer
    def forward(self, vertices, mask, _):
        B, N, D = vertices.size()
        messages = torch.cat([
            vertices.view(B, N, 1, D).expand(-1, -1, N, -1),
            vertices.view(B, 1, N, D).expand(-1, N, -1, -1),
        ], dim=3)
        messages = self.message_layer(messages)
        message_mask = mask.view(B, N, 1, 1) * mask.view(B, 1, N, 1)
        messages = messages.where(message_mask.expand_as(messages), torch.zeros_like(messages))
        vertices = torch.cat([
            vertices,
            messages.sum(dim=2),
        ], dim=2)
        vertices = self.node_layer(vertices)
        return vertices, None

class GNN(nn.ModuleList):
    def __init__(self, module_list):
        super().__init__(module_list)
    def forward(self, vertices, mask, vertex_states):
        """
        Args:
            vertices      (Tensor (B, N, D)): The graph nodes at a single point in time
            mask          (Tensor (B, N)): A mask marking which elements in the graph are active
            vertex_states (None or list of Tensor (B, N, D)): States at different layers
        """
        if vertex_states is None:
            vertex_states = [None for layer in self]
        else:
            vertex_states = vertex_states.copy()
        for idx, layer in enumerate(self):
            vertices, vertex_states[idx] = layer(vertices, mask, vertex_states[idx])
        return vertices, vertex_states


example_network = GNN(
    GraphBlock(
        message_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 64), nn.ReLU(inplace=True)),
        node_layer    = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 64), nn.ReLU(inplace=True))
    ),
    LSTMForGNN(64, 64),
    GraphBlock(
        message_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 64), nn.ReLU(inplace=True)),
        node_layer    = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 64), nn.ReLU(inplace=True))
    ),
    LSTMForGNN(64, 64),
)


class LinearPositionPredictor(nn.Module):
    def __init__(self, Din, Dspatial=2, Dtemporal=4):
        self.layer = nn.Linear(Din, Dspatial * Dtemporal)
    def forward(self, x):
        x = self.layer(x)
        return x
