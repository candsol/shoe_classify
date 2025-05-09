import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed: int=23):
    """
    Sets a specific seed for the tests
    
    Args:
        seed (int, optional): Random seed to set. Defaults is 23.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_func(model: nn.Module, data: torch.utils.data.DataLoader, loss_fn:nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device):
    train_loss, train_acc = 0, 0
    all_preds = []
    all_labels = []
    model.train()
    for bathc, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred = torch.softmax(y_logit, 1).argmax(1)
        train_acc += (y_pred == y).sum().item() / len (y_pred)
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    train_loss = train_loss / len(data)
    train_acc = train_acc / len(data)
    
    return train_loss, train_acc, all_preds, all_labels


def test_func(model: nn.Module, data: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            y_pred = y_logits.argmax(1)
            test_acc += (y_pred == y).sum().item() / len (y_pred)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss = test_loss / len(data)
    test_acc = test_acc / len(data)
    return test_loss, test_acc, all_preds, all_labels


def train(
        model: nn.Module, test_data: DataLoader, train_data: DataLoader, loss_fn:nn.Module,
        optimizer: torch.optim.Optimizer, device: torch.device, epochs: int =10
):
    set_seed(23)
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_func(
            model,train_data,loss_fn,optimizer
        , device)
        test_loss, test_acc = test_func(
            model, test_data, loss_fn
        , device)
        print(
            f"Epoch {epoch+1} |"
            f"train_loss :{train_loss: .4f} |"
            f"train_acc :{train_acc: .4f} |"
            f"test_loss :{test_loss: .4f} |"
            f"test_acc :{test_acc: .4f} "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results