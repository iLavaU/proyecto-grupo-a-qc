import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from config import Config
from utils.metrics import compute_metrics

n_qubits = 8
n_layers = 2
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
    def __init__(self, n_qubits=8):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

    def forward(self, x):
        device, dtype = x.device, x.dtype
        outputs = []
        for xi in x:
            q_out = quantum_circuit(xi, self.weights)
            q_out = torch.stack(q_out).to(device=device, dtype=dtype)
            outputs.append(q_out)
        return torch.stack(outputs)

class ConvResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)
    
class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + x)

class AsymmetricLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (salida de la red)
        targets: [B, H, W] o [B, 1, H, W] (enteros de clase)
        """
        num_classes = logits.shape[1]
        # One-hot encoding
        if targets.dim() == 3:
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        elif targets.dim() == 4 and targets.shape[1] == 1:
            targets_one_hot = F.one_hot(targets.squeeze(1).long(), num_classes).permute(0,3,1,2).float()
        else:
            targets_one_hot = targets.float()

        # Probabilidades con softmax
        probs = F.softmax(logits, dim=1)

        # True Positives / False Negatives / False Positives
        TP = (probs * targets_one_hot).sum(dim=(0,2,3))
        FN = ((1 - probs) * targets_one_hot).sum(dim=(0,2,3))
        FP = (probs * (1 - targets_one_hot)).sum(dim=(0,2,3))

        # Tversky index asimétrico
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Focal Tversky Loss
        loss = torch.pow(1 - tversky, self.gamma)

        return loss.mean()


class HybridQuantum(LightningModule):
    def __init__(self, config=Config(), in_ch=3, out_ch=10):
        super().__init__()
        self.config = config
        self.num_classes = out_ch
        self.train_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.train_acc = MulticlassAccuracy(num_classes=out_ch)
        self.val_acc = MulticlassAccuracy(num_classes=out_ch)
        self.test_acc = MulticlassAccuracy(num_classes=out_ch)
        self.test_preds = []
        self.test_targets = []

        self.qc = QuantumLayer(n_qubits=n_qubits)

        self.enc1 = ConvResBlock(in_ch, 8)
        self.enc2 = ConvResBlock(8, 16)
        self.enc3 = ConvResBlock(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.mlp1 = nn.Sequential(
            nn.Linear(32*4*4, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(128, n_qubits),
            nn.LayerNorm(n_qubits)
        )

        self.mlp3_1 = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.ReLU()
        )


        self.mlp4 = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.mlp6 = nn.Sequential(
            nn.Linear(128, n_qubits),
            nn.LayerNorm(n_qubits)
        )
        self.fc_expand = nn.Linear(n_qubits*3, 32*4*4)
        self.dec3 = ConvResBlock(32+32, 16)
        self.dec2 = ConvResBlock(16+16, 8)
        self.dec1 = nn.Conv2d(8, out_ch, 1)
        self.criterion =  AsymmetricLoss()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        B = x3.shape[0]
        x3_pooled = self.adaptive_pool(x3)
        x_flat = x3_pooled.view(B, -1)

        # ----- MLP encoder -----
        x1_m = self.mlp1(x_flat)
        x2_m = self.mlp2(x1_m)
        x3_m = self.mlp3(x2_m)

        # ----- Quantum -----
        x_q = self.mlp3_1(torch.tanh(x3_m))#self.qc(torch.tanh(x3_m) * torch.pi)

        # ----- MLP decoder -----
        x4_m = self.mlp4(x_q)
        x5_m = self.mlp5(x4_m)
        x6_m = self.mlp6(x5_m)

        # ----- Fusion -----
        x_cat = torch.cat([x3_m, x6_m, x_q], dim=1)
        x_latent = self.fc_expand(x_cat).view(B, 32, 4, 4)

        # ----- Decoder CNN -----
        x_up = F.interpolate(x_latent, size=x3.shape[2:], mode="nearest")
        x_up = self.dec3(torch.cat([x_up, x3], dim=1))

        x_up = F.interpolate(x_up, size=x2.shape[2:], mode="nearest")
        x_up = self.dec2(torch.cat([x_up, x2], dim=1))

        x_up = F.interpolate(x_up, size=x1.shape[2:], mode="nearest")
        return self.dec1(x_up)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_f1", self.train_f1(preds, y.squeeze(1)), prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc(preds, y.squeeze(1)), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1(preds, y.squeeze(1)), prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc(preds, y.squeeze(1)), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        metrics = compute_metrics(y_hat, y, self.num_classes)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_iou", metrics["IoU"], on_epoch=True)
        self.log("test_f1", metrics["F1 Score"], on_epoch=True)
        self.log("test_acc", metrics["Pixel Accuracy"], on_epoch=True)

    def on_test_epoch_end(self):
        self.test_preds = torch.cat(self.test_preds)
        self.test_targets = torch.cat(self.test_targets)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
