import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu()
            y_pred.append(preds)
            y_true.append(yb)
    import numpy as np
    y_true = np.concatenate([y.numpy() for y in y_true])
    y_pred = np.concatenate([y.numpy() for y in y_pred])
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm