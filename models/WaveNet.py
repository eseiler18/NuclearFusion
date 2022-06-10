from torch import nn
import numpy as np
import torch


class CausalDilatedConv1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation,
                 causal=True):
        super().__init__()
        self.causal = causal
        self.conv1D = nn.Conv1d(in_channel, out_channel, kernel_size,
                                dilation=dilation, bias=False, padding="same")
        if self.causal:
            self.ignoreIndex = (kernel_size-1)*dilation

    def forward(self, x):
        if self.causal:
            return self.conv1D(x)[..., :-self.ignoreIndex]
        else:
            return self.conv1D(x)


class ResidualBlock(nn.Module):
    def __init__(self, res_channel, skip_channel, kernel_size, dilation,
                 dropout):
        super().__init__()
        self.dilatedConv = CausalDilatedConv1D(res_channel, res_channel,
                                               kernel_size,
                                               dilation=dilation)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.resConv1D = nn.Conv1d(res_channel, res_channel,
                                   kernel_size=1)
        self.skipConv1D = nn.Conv1d(res_channel, skip_channel,
                                    kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip_size):
        x1 = self.dilatedConv(x)
        x1 = self.relu(x1)*self.tanh(x1)
        x1 = self.dropout(x1)
        resOutput = self.resConv1D(x1)
        resOutput = resOutput + x[..., -resOutput.size(2):]
        skipOutput = self.skipConv1D(x1)
        skipOutput = skipOutput[..., -skip_size:]
        return resOutput, skipOutput


class StackResidualBloc(nn.Module):
    def __init__(self, res_channel, skip_channel, kernel_size, stack_size,
                 layer_size, dropout):
        super().__init__()
        allDilations = self.buildDilation(stack_size, layer_size)
        self.resBlocs =  nn.ModuleList([])
        for dilations in allDilations:
            for dilation in dilations:
                self.resBlocs.append(ResidualBlock(res_channel, skip_channel,
                                                   kernel_size, dilation,
                                                   dropout))

    def forward(self, x, skip_size):
        resOutput = x
        allSkipOutput = []
        for resBloc in self.resBlocs:
            resOutput, skipOutput = resBloc(resOutput, skip_size)
            allSkipOutput.append(skipOutput)
        return resOutput, torch.stack(allSkipOutput)

    @staticmethod
    def buildDilation(stack_size, layer_size):
        dilationCoeff = [2**i for i in range(10)]
        allDilations = []
        for i in range(stack_size):
            dilations = []
            for j in range(layer_size):
                dilations.append(dilationCoeff[j % 10])
            allDilations.append(dilations)
        return allDilations


class DenseLayer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1D = nn.Conv1d(in_channel, in_channel, kernel_size=1,
                                bias=False)
        self.relu = nn.ReLU()

    def forward(self, allSkipOutput):
        # dim stack/batch/channel/time sample
        x = allSkipOutput.mean(dim=0)
        for i in range(2):
            x = self.relu(x)
            x = self.conv1D(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, channel, kernel_size, stack_size,
                 layer_size, seqLen, dropout):
        super().__init__()
        self.layerSize = layer_size
        self.stackSize = stack_size
        self.kernelSize = kernel_size
        self.skipSize = int(seqLen)-self.calculateReceptiveField()
        if self.skipSize <= 0:
            print("skip size must be positive value : ", self.skipSize)
            print("decrease layer/stack size or increase seqLen")
        self.causalConv = CausalDilatedConv1D(channel, channel,
                                              kernel_size, dilation=1)
        self.resBlocks = StackResidualBloc(channel, channel,
                                           kernel_size, self.stackSize,
                                           self.layerSize, dropout)
        self.denseLayer = DenseLayer(channel)

    def calculateReceptiveField(self):
        out = [(self.kernelSize - 1) * (2 ** i) for i in range(self.layerSize)]
        out = out*self.stackSize
        return np.sum(out) + self.kernelSize

    def forward(self, x):
        x = self.causalConv(x)
        _, skipConnections = self.resBlocks(x, self.skipSize)
        output = self.denseLayer(skipConnections)
        return output


class WaveNetClassifier(nn.Module):
    def __init__(self, channel, kernel_size, stack_size,
                 layer_size, seqLen, output_size, dropout):
        super().__init__()
        self.output_size = output_size
        self.waveNet = WaveNet(channel, kernel_size, stack_size, layer_size,
                               seqLen, dropout)
        self.linear = nn.Linear(channel, self.output_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.waveNet(x)
        x = self.linear(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        output = self.softmax(x)
        return output
