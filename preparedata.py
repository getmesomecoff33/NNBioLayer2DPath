import os
import PIL
import json
import numpy
import torch
import torchvision

from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DEBUG = True

with open("config.json") as file:
    config = json.load(file)
    DATAPATH = config["DataPath"]
    RAWDATA = config["RawData"]
SUBSETS = ["test","train", "validate"]

dataDir = os.path.join(DATAPATH,RAWDATA)

files =  os.listdir(dataDir)

classes = []
for file in files:
    name = file.split("_")[-1].split(".")[0]
    classes.append(name)

for sub in SUBSETS:
    tmp = os.path.join(DATAPATH, sub)
    if not os.path.exists(tmp):
        os.makedirs(tmp)

def create_Dataset():
    for file in files:
        dataset = numpy.load(os.path.join(dataDir, file))
        numpy.random.shuffle(dataset)
        dataset = dataset[:35000]
        test, val, train = numpy.split(dataset,[int(0.2*len(dataset)), int(0.3*len(dataset))])
        numpy.save(os.path.join(DATAPATH, SUBSETS[0],file),test)
        numpy.save(os.path.join(DATAPATH, SUBSETS[1],file),train)
        numpy.save(os.path.join(DATAPATH, SUBSETS[2],file),val)
        pass

def generate_Dateset(dataType):
    transform_norm = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), 
    torchvision.transforms.Resize((28,28)),torchvision.transforms.RandomRotation(10)])
    dataDir = os.path.join(DATAPATH,dataType)
    files = os.listdir(dataDir)
    classes = []
    for file in files:
        name = file.split("_")[-1].split(".")[0]
        classes.append(name)

    classes = sorted(classes)
    classesIndex = {classes[i]: i for i in range(len(classes))}
    if DEBUG:
        print(classes)
    dataset = {}
    dataX = []
    dataY = []
    for file in files:
        name = file.split("_")[-1].split(".")[0]
        x = numpy.load(os.path.join(dataDir, file)).reshape(-1, 28, 28) / 255
        y = numpy.ones((len(x),1),dtype=numpy.int64) * classesIndex[name]
        dataX.extend(x)
        dataY.extend(y)
    yTorchStack = torch.stack([torch.Tensor(j) for j in dataY])

    xTorchStack = torch.stack([transform_norm(Image.fromarray(numpy.uint8(i*255))) for i in dataX]) 

    return torch.utils.data.TensorDataset(xTorchStack, yTorchStack)


if __name__ == "__main__":
    #prep_Data()
    et = numpy.load(os.path.join(DATAPATH, SUBSETS[1],"full_numpy_bitmap_mushroom.npy"))
    train = generate_Dateset('train')
    test = generate_Dateset('test')
    val = generate_Dateset('validate')
    torch.save(train,'tensortrain.pt')
    torch.save(test,'tensortest.pt')
    torch.save(val,'tensorval.pt')
    print(et.shape)
    