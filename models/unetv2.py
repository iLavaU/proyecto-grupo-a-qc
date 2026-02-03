import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

n_qubits = 8
n_layers_conv = 2
n_layers_pool = 1
dev = qml.device("default.qubit", wires=n_qubits)

def conv_block(params, wires):
    for i in range(len(wires)-1):
        qml.RZ(params[i,0], wires=wires[i])
        qml.RY(params[i,1], wires=wires[i+1])
        qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.RY(params[i,2], wires=wires[i+1])

def pool_block(params, sources, sinks):
    for s, t in zip(sources, sinks):
        qml.RZ(params[0], wires=s)
        qml.RY(params[1], wires=t)
        qml.CNOT(wires=[s,t])
        qml.RY(params[2], wires=t)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, conv_params, pool_params):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(n_layers_conv):
        conv_block(conv_params[l], range(n_qubits))
    pool_block(pool_params[0], [0,1,2,3], [4,5,6,7])
    pool_block(pool_params[1], [0,1], [2,3])
    pool_block(pool_params[2], [0], [1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_params = nn.Parameter(0.01 * torch.randn(n_layers_conv, n_qubits-1,3))
        self.pool_params = nn.Parameter(0.01 * torch.randn(3,3))
    def forward(self, x):
        q_out = []
        for xi in x:
            q_out.append(torch.tensor(quantum_circuit(xi, self.conv_params, self.pool_params), dtype=torch.float32))
        return torch.stack(q_out)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class HybridUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,8)
        self.enc2 = ConvBlock(8,16)
        self.enc3 = ConvBlock(16,32)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2,2))
        self.qc = QuantumLayer()
        self.fc_expand = nn.Linear(n_qubits,32*2*2)
        self.dec3 = ConvBlock(32+32,16)
        self.dec2 = ConvBlock(16+16,8)
        self.dec1 = nn.Conv2d(8,out_ch,1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        B,C,H,W = x3.shape
        x3p = self.adaptive_pool(x3)
        x_flat = x3p.view(B,-1)[:,:n_qubits]
        x_q = self.qc(x_flat)
        x_latent = self.fc_expand(x_q).view(B,32,2,2)
        x_up = F.interpolate(x_latent,size=x3.size()[2:], mode='nearest')
        x_up = torch.cat([x_up,x3],dim=1)
        x_up = self.dec3(x_up)
        x_up = F.interpolate(x_up,size=x2.size()[2:], mode='nearest')
        x_up = torch.cat([x_up,x2],dim=1)
        x_up = self.dec2(x_up)
        x_up = F.interpolate(x_up,size=x1.size()[2:], mode='nearest')
        out = self.dec1(x_up)
        return torch.sigmoid(out)
