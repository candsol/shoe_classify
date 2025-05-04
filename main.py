# from src.utils.normalización import normalice_and_save
from src.utils.data_load import load_data
from src.utils.experiment_generator import run_experiments
import json

def pipeline(output_folder, inference_path):
    with open("parameters.json", "r",encoding="utf-8") as file:
        parameters = json.load(file)

    AUGMENTATION = True
    ALL_DATA = False
    WEIGHTED_LOSS = False

    if "no_augmentation" in parameters:
        AUGMENTATION = False
        WEIGHTED_LOSS = True

    if "no_all_data" in parameters:
        ALL_DATA = False
        WEIGHTED_LOSS = True

    print("[INFO] Creando dataloaders")
    train_loader, test_loader = load_data(output_folder, parameters["batches"],
                                          augmentation=AUGMENTATION, all_data=ALL_DATA)
    print("[INFO] Dataloaders creados")

    
    print("[INFO] Iniciando entrenamiento")
    run_experiments(test_loader, train_loader, parameters, w_loss=False)
    print("[INFO] Entrenamiento finalizado")

    return


if __name__ == "__main__":
    OUTPUT_PATH = "output/"
    INFERENCE_PATH = "inference/"

    pipeline(OUTPUT_PATH, INFERENCE_PATH)
