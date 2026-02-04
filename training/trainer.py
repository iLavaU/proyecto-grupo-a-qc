"""
Training and testing utilities using PyTorch Lightning
"""

import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score
from utils.metrics import compute_metrics  # <- importamos tus métricas personalizadas


def train_model(model, train_loader, val_loader, config):
    """
    Train the hybrid quantum classifier.
    """
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Encoding: {config.ENCODING}")
    print(f"Circuit: {config.CIRCUIT_TYPE}")
    print(f"Device: {config.DEVICE}")
    print(f"Qubits: {config.N_QUBITS}")
    print(f"Layers: {config.N_LAYERS}")
    print(f"Max Epochs: {config.MAX_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 60 + "\n")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        mode='min',
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_acc:.4f}',
        verbose=True
    )

    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False
    )

    print("Starting training...")
    start_time = time.time()

    try:
        trainer.fit(model, train_loader, val_loader)
        training_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        training_time = time.time() - start_time
        print(f"Training time before interruption: {training_time:.2f} seconds")

    return model, trainer, training_time


def test_model(model, trainer, test_loader, class_names):
    """
    Test the trained model on the test set using custom metrics.
    """
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)

    # Run testing
    test_results = trainer.test(model, test_loader)

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(model.device), y.to(model.device)
            y_hat = model(x)
            all_preds.append(y_hat)
            all_targets.append(y)

    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)

    # Métricas personalizadas
    metrics = compute_metrics(preds_tensor, targets_tensor, model.num_classes)

    # Accuracy y F1 con sklearn
    preds_np = torch.argmax(preds_tensor, dim=1).cpu().numpy().flatten().astype(int)
    targets_np = targets_tensor.cpu().numpy().flatten().astype(int)

    test_acc = accuracy_score(targets_np, preds_np)
    test_f1 = f1_score(targets_np, preds_np, average='weighted')

    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Test IoU: {metrics['IoU']:.4f}")
    print("=" * 60 + "\n")

    return {
        'test_results': test_results,
        'predictions': preds_np,
        'targets': targets_np,
        'accuracy': test_acc,
        'f1_score': test_f1,
        'iou': metrics['IoU'],
        'class_names': class_names
    }


def save_model(model, config, class_names, test_accuracy, test_f1, test_iou, training_time, filename=None):
    """
    Save the trained model and metadata.
    """
    if filename is None:
        filename = f"hybrid_quantum_{config.ENCODING}_{config.CIRCUIT_TYPE}.pth"

    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config.get_config_dict(),
        'class_names': class_names,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_iou': test_iou,
        'training_time': training_time
    }

    torch.save(save_dict, filename)
    print(f"\nModel saved as: {filename}")
    return filename


def load_model(filename, model_class, quantum_device):
    """
    Load a saved model.
    """
    checkpoint = torch.load(filename)
    print(f"Model loaded from: {filename}")
    print(f"Test accuracy: {checkpoint['test_accuracy']:.4f}")
    print(f"Test F1-score: {checkpoint['test_f1']:.4f}")
    print(f"Test IoU: {checkpoint.get('test_iou', 0):.4f}")
    print(f"Training time: {checkpoint['training_time']:.2f} seconds")
    return checkpoint


def print_training_summary(model, test_results, training_time, config):
    """
    Print a comprehensive summary of training results.
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Encoding: {config.ENCODING}")
    print(f"  Circuit: {config.CIRCUIT_TYPE}")
    print(f"  Qubits: {config.N_QUBITS}")
    print(f"  Layers: {config.N_LAYERS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")

    print("\nPerformance:")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"  Test F1-score: {test_results['f1_score']:.4f}")
    print(f"  Test IoU: {test_results['iou']:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("=" * 60 + "\n")
