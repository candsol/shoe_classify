import os
import json
import torch
from torch.utils.data import DataLoader
from torch import nn

from src.utils.normalización import normalized_dataset
from src.utils.engine import set_seed
from src.utils.model_generator import TinyVGG
from src.utils.model_generator import get_model
from src.utils.summary_utils import select_optimizer, create_write, train



def run_experiments(test_dataloader: DataLoader, train_dataloader: DataLoader, parameters: dict, w_loss= False):

    torch.cuda.empty_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()


    experiment_name = parameters["name"]
    num_epochs = parameters["epochs"]
    optimizers = parameters["optimizers"]
    experiment_number = 0
    if w_loss:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0])).to(device) #El doble de peso para los 0
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)
    conf_mat = []

    for model_name in parameters["models"]:
        best_test_acc = 0
        best_checkpoint = None
        best_test_loss = 1000
        for epochs in num_epochs:
            for optimizer_name in optimizers:
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] model: {model_name}")
                print(f"[INFO] Optimizer: {optimizer_name}")
                print(f"[INFO] Epochs: {epochs}")
                model = get_model(model_name).to(device)
                writer = create_write(name=optimizer_name,
                model=model_name, experiment_name=experiment_name, extra=str(epochs))
                #log_dir = os.path.join("log", timestamp + optimizer + name + str(epochs))
                optimizer = select_optimizer(model, optimizer_name)
                results, best_model, test_acc, cm_fig, test_loss = train(model, test_dataloader, train_dataloader, loss_fn, optimizer, device,
                epochs, writer, experiment_name + "/" + model_name + "/" + optimizer_name + "/" + str(epochs))
                print("-" * 50 + "\n")

                if test_loss < best_test_loss:
                    best_test_acc = test_acc
                    best_checkpoint = best_model
                    best_optimizer = optimizer_name
                    best_epochs = epochs
                    best_model_name = model_name
                    best_test_loss = test_loss
                
            conf_mat.append(cm_fig)
        if best_checkpoint:
            print(f"[INFO] Best model: {best_model_name} with optimizer: {best_optimizer} and epochs: {best_epochs}")
            path = f"./models/{experiment_name}/{best_model_name}/{best_optimizer}/{best_epochs}"
            os.makedirs(path, exist_ok=True)
            torch.save(best_checkpoint, path + f"/best_model_{best_model_name}.pth")
            

    torch.cuda.empty_cache()
    return conf_mat