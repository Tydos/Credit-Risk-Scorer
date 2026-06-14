import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


def train_model(
    model,
    trainerloader,
    valloader,
    testloader,
    optimizer,
    criterion,
    EPOCH,
    device,
    mlflow,
):
    loss_history = []
    val_loss_history = []
    test_loss_history = []

    model.train()
    for epoch in tqdm(range(EPOCH), desc="Epochs"):
        total_loss = 0

        # --- Training loop ---
        for features, labels in trainerloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(trainerloader)
        loss_history.append(avg_train_loss)

        # --- Validation loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in valloader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_features).squeeze(-1)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(valloader)
        val_loss_history.append(avg_val_loss)

        # --- Test evaluation ---
        test_loss = 0
        with torch.no_grad():
            for test_features, test_labels in testloader:
                test_features = test_features.to(device)
                test_labels = test_labels.to(device)

                test_outputs = model(test_features).squeeze(-1)
                test_loss += criterion(test_outputs, test_labels).item()

        avg_test_loss = test_loss / len(testloader)
        test_loss_history.append(avg_test_loss)

        # --- AUC and F1 on validation set ---
        all_labels, all_probs = [], []
        with torch.no_grad():
            for vf, vl in valloader:
                probs = torch.sigmoid(model(vf.to(device)).squeeze(-1))
                all_probs.append(probs.cpu().numpy())
                all_labels.append(vl.numpy())
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        val_preds = (all_probs >= 0.5).astype(int)
        val_auc       = roc_auc_score(all_labels, all_probs)
        val_f1        = f1_score(all_labels, val_preds)
        val_accuracy  = accuracy_score(all_labels, val_preds)
        val_precision = precision_score(all_labels, val_preds)
        val_recall    = recall_score(all_labels, val_preds)

        # --- Log metrics per epoch ---
        mlflow.log_metric("train_loss",    avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss",      avg_val_loss,   step=epoch)
        mlflow.log_metric("test_loss",     avg_test_loss,  step=epoch)
        mlflow.log_metric("val_auc",       val_auc,        step=epoch)
        mlflow.log_metric("val_f1",        val_f1,         step=epoch)
        mlflow.log_metric("val_accuracy",  val_accuracy,   step=epoch)
        mlflow.log_metric("val_precision", val_precision,  step=epoch)
        mlflow.log_metric("val_recall",    val_recall,     step=epoch)


        logging.info(
            f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f} | "
            f"Val Loss={avg_val_loss:.4f}, AUC={val_auc:.4f}, "
            f"F1={val_f1:.4f}, Acc={val_accuracy:.4f}, "
            f"Prec={val_precision:.4f}, Rec={val_recall:.4f}"
        )

        model.train()

    return model, loss_history, val_loss_history, test_loss_history
