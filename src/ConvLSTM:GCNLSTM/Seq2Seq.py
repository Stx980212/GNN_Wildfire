import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, out_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, return_sequences=False, variable_len=False):

        super(Seq2Seq, self).__init__()
        
        self.variable_len = variable_len

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )
        
        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 
        
        self.return_sequences=return_sequences

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding)
        if return_sequences:
            self.conv_sequence = nn.Conv3d(
                in_channels=num_kernels, out_channels=num_channels,
                kernel_size=tuple([kernel_size[0]]*3), padding=tuple([padding[0]]*3))

    def forward(self, X, Len=None):
        
        if self.variable_len:
            # Forward propagation through all the layers
            output = self.sequential(X)
            if self.return_sequences:
                raise NotImplementedError
            else:
                output = torch.gather(output,2,(Len-1).view(-1, 1, 1, 1, 1)
                 .expand(output.size(0), output.size(1),
                         1, output.size(3), output.size(4))).squeeze(2)
                output = self.conv(output)
        else:
            output = self.sequential(X)
            if self.return_sequences:
                output=self.conv_sequence(output)
            else:
                # Return only the last output frame
                output = self.conv(output[:,:,-1])

        return nn.Sigmoid()(output)
