import torch
from torch import nn
from efficientnet_pytorch_3d import EfficientNet3D
import torchvision
from src.utils.engine import set_seed

def get_model(model_name: str) -> nn.Module:

    if model_name == "tinyvgg":
        model = TinyVGG(input_shape=1, hidden_units=64, output_shape=2)
        return model
    
    if "custom" in model_name:
        model_name = model_name.split("_")[1]
        if model_name == "effntb0":
            model = create_effntb0()
            model = CustomEfficientNet(model)
            return model
        if model_name == "effntb2":
            model = create_effntb2()
            model = CustomEfficientNet(model)
            return model
        if model_name == "convnextTiny":
            model = create_convnext_tiny()
            model = CustomEfficientNet(model)
            return model
        if model_name == "convnextSmall":
            model = create_convnext_small()
            model = CustomEfficientNet(model)
            return model
        if model_name == "resnet50":
            model = create_resnet50()
            model = CustomEfficientNet(model)
            return model
        if model_name == "resnet101":
            model = create_resnet101()
            model = CustomEfficientNet(model)
            return model
        if model_name == "resnet152":
            model = create_resnet152()
            model = CustomEfficientNet(model)
            return model

    
    model = EfficientNet3D.from_name(f"{model_name}", override_params={'num_classes': 2}, in_channels=1)
    return model
        


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.LayerNorm((7683200,), elementwise_affine=False, eps=1e-5),
            nn.Linear(in_features= 7683200,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class PreprocessInput(nn.Module):
    def __init__(self):
        super(PreprocessInput, self).__init__()

        self.conv_block_1 = nn.Sequential(
        nn.Conv3d(1, 100, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.Conv3d(100, 50, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=1)
        )

        self.conv_block_2 = nn.Sequential(
        nn.Conv3d(50, 25, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.Conv3d(25, 13, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=1)
        )

        self.conv_block_3 = nn.Sequential(
        nn.Conv3d(13, 3, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU()
        )

        self.flatten = nn.Flatten(start_dim=3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((244, 244))


    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x)  
        x = self.conv_block_3(x)  
        x = self.flatten(x)  
        x = self.adaptive_pool(x)  
        return x

class CustomEfficientNet(nn.Module):
    def __init__(self, base_model):
        super(CustomEfficientNet, self).__init__()
        self.preprocess = PreprocessInput()  # Capa de preprocesamiento
        self.model = base_model  # Modelo EfficientNet

    def forward(self, x):
        x = self.preprocess(x)  # Preprocesar entrada
        x = self.model(x)  # Pasar por EfficientNet
        return x

def create_effntb0() -> nn.Module: 
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    model = freeze_parameters(model)
    print("[INFO] create new effntb0 model.")
    return model

def freeze_parameters(model: nn.Module) -> nn.Module:
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    )
    return model

def create_effntb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=1408, out_features=2, bias=True)
    )
    print("[INFO] create new effntb2 model.")
    return model

def create_convnext_tiny():
    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    model = torchvision.models.convnext_tiny(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=2, bias=True)
    )
    print("[INFO] create new convnext_tiny model.")
    return model

def create_convnext_small():
    weights = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
    model = torchvision.models.convnext_small(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=2, bias=True)
    )
    print("[INFO] create new convnext_small model.")
    return model

def create_resnet50():
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
    )
    print("[INFO] create new resnet50 model.")
    return model

def create_resnet101():
    weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet101(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
    )
    print("[INFO] create new resnet101 model.")
    return model

def create_resnet152():
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet152(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
    )
    print("[INFO] create new resnet152 model.")
    return model