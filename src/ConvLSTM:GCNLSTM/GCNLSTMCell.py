import torch
import torch.nn as nn
from GCNLayer import GCNLayer

# Original GCNLSTM cell as proposed by Shi et al.
class GCNLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    activation, frame_size):

        super(GCNLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.GCN = nn.GCNLayer(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            )           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.rand(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.rand(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.rand(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        GCN_output = self.GCN(torch.cat([X, H_prev], dim=1))

        i_GCN, f_GCN, C_GCN, o_GCN = torch.chunk(GCN_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_GCN + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_GCN + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_GCN)

        output_gate = torch.sigmoid(o_GCN + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
