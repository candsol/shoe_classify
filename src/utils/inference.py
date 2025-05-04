# from utils.model_generator import get_model
# import torch
# import json
# import os
# import nibabel as nib
# import numpy as np
# from utils.normalización import normalice_and_save, search_paths
# from torch.utils.data import DataLoader
#
# class InferenceData:
#     def __init__(self, file_paths):
#         self.file_paths = file_paths
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.file_paths[idx]
#         image = nib.load(image_path).get_fdata()
#         image = image.astype(np.float32)
#         image = np.expand_dims(image, axis=0)
#         image = torch.from_numpy(image)
#         return image, image_path
#
#
#
# def inference_data(inference_path):
#
#     normalice_and_save(inference_path)
#     files = search_paths(inference_path, "brain_normalized.nii.gz")
#
#     weights_path = 'models/Test_4_New_Metrics/custom_resnet152/Adam/150/best_model_custom_resnet152.pth'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     weights = torch.load(weights_path)
#     model = get_model('custom_resnet152').to(device)
#     model.load_state_dict(weights)
#     inference_dict = {}
#     inference_data = InferenceData(files)
#     inference_data = DataLoader(inference_data, batch_size=len(files), shuffle=False)
#
#
#     for image, path in inference_data:
#         model.eval()
#         with torch.inference_mode():
#
#             images = image.to(device)
#             output = model(images)
#             logits = torch.softmax(output, dim=1)
#             inferences = torch.argmax(logits, dim=1)
#
#             for path, logit, inference in zip(path, logits, inferences):
#                 file_name = path.split("/")[-1]
#                 inference_dict[file_name] = {
#                     "logits": logit.cpu().detach().numpy().tolist(),
#                     "inference": inference.cpu().detach().numpy().tolist()
#                 }
#
#     with open('inference/labels.json', 'w',encoding="utf") as f:
#         json.dump(inference_dict, f, indent=4)