import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from segmentation_models_pytorch.encoders import get_encoder

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
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

    def forward(self, x):
        device, dtype = x.device, x.dtype
        outputs = []
        for xi in x:
            q_out = quantum_circuit(xi, self.weights)
            q_out = torch.stack(q_out).to(device=device, dtype=dtype)
            outputs.append(q_out)
        return torch.stack(outputs)

class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + x)

class MITFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_encoder("mit_b0", weights="imagenet")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = self.encoder.out_channels[-1]
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)[-1]
            pooled = self.avgpool(features)
            return pooled.view(x.size(0), -1)

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
    def __init__(self, config=None, out_ch=10, target_size=(640, 640)):
        super().__init__()
        self.config = config
        self.num_classes = out_ch
        self.target_size = target_size

        self.train_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=out_ch, average="macro")
        self.train_acc = MulticlassAccuracy(num_classes=out_ch)
        self.val_acc = MulticlassAccuracy(num_classes=out_ch)
        self.test_acc = MulticlassAccuracy(num_classes=out_ch)

        self.encoder = MITFeatureExtractor()

        self.mlp_pre_q = nn.Sequential(
            nn.Linear(self.encoder.out_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.ReLU()
        )
        self.mlp_post_q = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.ReLU()
        )
        self.qc = QuantumLayer(n_qubits=n_qubits)

        self.decoder_start_size = 8 
        self.decoder_channels = 64

        self.decoder_fc = nn.Linear(n_qubits, self.decoder_channels * self.decoder_start_size**2)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_channels, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_ch, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Upsample(size=self.target_size, mode='bilinear', align_corners=False) # al tamaño final
        )

        # Loss
        self.criterion = AsymmetricLoss()

    def forward(self, x):
        B = x.size(0)
        feats = self.encoder(x)
        x_mlp = F.relu(self.mlp_pre_q(feats))
        x_q = self.qc(torch.tanh(x_mlp) * torch.pi)
        x_post = self.mlp_post_q(x_q)

        x_dec = self.decoder_fc(x_post)
        x_dec = x_dec.view(B, self.decoder_channels, self.decoder_start_size, self.decoder_start_size)
        x_out = self.decoder_conv(x_dec)
        return x_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1).view(-1)
        y_flat = y.view(-1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.train_f1(preds, y_flat), prog_bar=True)
        self.log("train_acc", self.train_acc(preds, y_flat), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1).view(-1)
        y_flat = y.view(-1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1(preds, y_flat), prog_bar=True)
        self.log("val_acc", self.val_acc(preds, y_flat), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, self.num_classes)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_iou", metrics["IoU"])
        self.log("test_f1", metrics["F1 Score"])
        self.log("test_acc", metrics["Pixel Accuracy"])

    def configure_optimizers(self):
        lr = getattr(self.config, "LEARNING_RATE", 1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
