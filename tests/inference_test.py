import torch
from PIL import Image
from matplotlib import pyplot as plt

from data import FloodNetDataset
from models.unetv1 import HybridUNet
from config import Config

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

config = Config()

#model = HybridUNet(config, 3, 10)

model_checkpoint = 'trained_models/best-model-epoch=41-val_acc=0.7087.ckpt'
image_path = 'FloodNet/FloodNet-Supervised_v1.0/val/val-org-img/6345.jpg'

model = HybridUNet.load_from_checkpoint(model_checkpoint, device=device)

model.eval()
model.freeze()
model = model.to(device)


image = Image.open(image_path).convert("RGB")

transform = FloodNetDataset.get_image_transform(image_size=config.IMAGE_SIZE)
transformed = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(transformed)

mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("Imagen"); plt.imshow(image); plt.axis("off")
plt.subplot(1,2,2); plt.title("Máscara"); plt.imshow(config.COLOR_MAP[mask]); plt.axis("off")
print()
plt.show()
