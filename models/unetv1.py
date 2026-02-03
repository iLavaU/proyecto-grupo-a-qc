import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pytorch_lightning import LightningModule
from config import Config


n_qubits = 4
n_layers = 1
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        q_out = []
        for xi in x:
            qc_out = quantum_circuit(xi, self.weights)
            qc_out = torch.stack(qc_out).to(device=device, dtype=dtype)
            q_out.append(qc_out)
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

class HybridUNet(LightningModule):
    def __init__(self, config=Config(), in_ch=3, out_ch=10):
        super().__init__()

        self.config=config

        # Storage for validation and test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.enc1 = ConvBlock(in_ch, 8)
        self.enc2 = ConvBlock(8, 16)
        self.enc3 = ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.qc = QuantumLayer()
        self.fc_latent = nn.Linear(32*4*4, n_qubits)
        self.fc_expand = nn.Linear(n_qubits, 32*4*4)
        self.dec3 = ConvBlock(32+32, 16)
        self.dec2 = ConvBlock(16+16, 8)
        self.dec1 = nn.Conv2d(8, out_ch, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        B,C,H,W = x3.shape
        x3_pooled = self.adaptive_pool(x3)
        x_flat = x3_pooled.view(B,-1)
        x_qin = self.fc_latent(x_flat)
        x_qout = self.qc(x_qin)
        x_latent = self.fc_expand(x_qout).view(B,32,4,4)

        x_up = F.interpolate(x_latent, size=x3.size()[2:], mode='nearest')
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.dec3(x_up)

        x_up = F.interpolate(x_up, size=x2.size()[2:], mode='nearest')
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.dec2(x_up)

        x_up = F.interpolate(x_up, size=x1.size()[2:], mode='nearest')
        out = self.dec1(x_up)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.validation_step_outputs.append({"preds": preds, "targets": y})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append({"preds": preds, "targets": y})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
