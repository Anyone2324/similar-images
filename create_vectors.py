import torchvision.models as models
import torch
from PIL import Image
import torchvision.transforms as transforms
from skimage.io import imread_collection
import json
from os import listdir
from os.path import isfile, join

loader = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image


model = models.resnet50(pretrained=True)
modelWithoutHead = torch.nn.Sequential(*(list(model.children())[:-1]))

allImages = [f for f in listdir("data/") if isfile(join("data/", f))]
print(modelWithoutHead(image_loader("data/" + allImages[0]))[0])


for i in range(len(allImages)):
    with open("results_test/" + (allImages[i])[:-4] + '.json', 'w') as f:
        vector = modelWithoutHead(image_loader("data/" + allImages[i]))
        toJson = {"name": allImages[i], "vector": vector[0].tolist()}
        print(toJson)
        json.dump(toJson, f)
