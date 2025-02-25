import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        return self.model(x)



def load_model(weights_path="model_V2.pt"):
    # Load the TorchScript model directly
    model = torch.jit.load(weights_path, map_location=torch.device("cpu"))
    model.eval()
    return model

'''
def load_model(weights_path="model_V2.pt"):
    model = PlantDiseaseModel(num_classes=38)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False))
    model.eval()
    return model
'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)