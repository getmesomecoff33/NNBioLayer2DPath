import os
import io
import json
import numpy
import torch
import random
import torch.utils
import torch.utils.data
import torchvision
import matplotlib.pyplot as pyplot

from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from itertools import islice
from netzmodelle import BioMLP2D

from makescreenshot import load_image
#from imageToWall import pipline_entry

SHOWFIGURE = True
SAVEFIGURE = True

DEBUG = False
SENDTOWALL = False
LOADFROMDATASET = False

SEED = 1
numpy.random.seed(SEED)
torch.manual_seed(SEED)

with open("config.json") as file:
    config = json.load(file)
    IMAGE_DIR = config["ImageDir"]
    IMAGE_NAME = config["ImageName"]
    RAWDATA = config["RawData"]
    DATAPATH = config["DataPath"]


DATADIR = os.path.join(DATAPATH,RAWDATA)

BATCHSIZE = 50
SHUFFLE = True


IMAGETRANSFORMER = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28,28))])

DATASIZE = 60000
STEPS = 40000

IMAGEWIDTH = 192
IMAGEHEIGHT = 112

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#Color Coding
CMAP = pyplot.colormaps.get_cmap('jet')
LINEWIDTH = 1

CLASSES = []

def plot_to_array(fig):
    fig.canvas.draw()
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = numpy.reshape(numpy.frombuffer(io_buf.getvalue(), dtype=numpy.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def load_Data(subset):
    if subset == "train":
        return torch.load("tensortrain.pt")
    if subset == "val":
        return torch.load("tensorval.pt")
    return torch.load("tensortest.pt")
    

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def accuracy(network, dataset, device, N=2000, batch_size=BATCHSIZE):
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    total = 0
    correct = 0
    for x, labels in islice(dataset_loader, N // batch_size):
        labels = labels.squeeze()
        logits = network(x.to(device))
        predicted_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(predicted_labels == labels.to(device))
        total += x.size(0)

    return correct/total

def loss_Function(network, dataset, device, N=2000, batch_size=BATCHSIZE):
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    loss = 0
    total = 0
    for x, labels in islice(dataset_loader, N // batch_size):
        labels = labels.squeeze()
        logits = network(x.to(device))
        loss += torch.sum((logits-torch.eye(10,)[labels.long()])**2)
        total += x.size(0)
    return loss/total

def get_classes(dataDir):
    classes = []
    files = os.listdir(dataDir) 
    for file in files:
        name = file.split("_")[-1].split(".")[0]
        classes.append(name)
    classes = sorted(classes)
    classesIndex = { i:classes[i] for i in range(len(classes))}
    return classesIndex

def load_image_from_dataset(dataset):
    index = random.randint(0, len(dataset))
    image, label = dataset[index]
    return image.squeeze().unsqueeze_(0), int(label)

    
def eval_image(model, img_normalized):    
    with torch.no_grad():
        model.eval()  
        logits = model(img_normalized.to(DEVICE))
        labelTensor = torch.argmax(logits, dim=1)
        label = int(labelTensor[0])
        certainty = logits[0][label]
        return label, certainty
    
## INIT DRAWING CANVAS AND DRAWING RELATED STUFF
def draw_connection_in_image(image, nodeLayer, layerNode, layerPlusNode,color):
    layerCentersInImage =[1,33,65,97,129,162,191] 
    layerPixel = layerCentersInImage[nodeLayer]
    nextLayerPixel = layerCentersInImage[nodeLayer+1]
    convertedColor = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
    layerOffset = 6
    nextlayerOffset = 6
    if nodeLayer == 5:
        nextlayerOffset = 11
    lineShape = [(layerPixel,int(layerNode)+layerOffset),(nextLayerPixel,int(layerPlusNode)+nextlayerOffset)]
    imgDraw = ImageDraw.Draw(image)
    imgDraw.line(lineShape,fill=convertedColor, width=0)
    return image


def init_image():
    #The base image, shows allways
    layerDistancePixel = 32
    layerCentersInImage =[1,33,65,97,129,162]
    image = Image.new('RGB', (IMAGEWIDTH, IMAGEHEIGHT))
    imageDraw = ImageDraw.Draw(image)
    for layer in layerCentersInImage:
        layerShapeCenter = [(layer,6), (layer,106)]
        layerShapeOutline = [(layer-1,6), (layer+1,106)]
        imageDraw.rectangle(layerShapeOutline,fill="white")
        imageDraw.rectangle(layerShapeCenter,fill="dimgrey")
    # Last Layer
    outNodes = [11,21,31,41,51,61,71,81,91,101]
    for nodePixel in outNodes:
        layerShapeOutline = [(190,nodePixel-3), (192,nodePixel+3)]
        layerShapeINline = [(190,nodePixel-2), (192,nodePixel+2)]
        imageDraw.rectangle(layerShapeOutline,fill="white")
        imageDraw.rectangle(layerShapeINline,fill="dimgrey")
    return image

def render_image_path(model,img_normalizied,step):
    image = init_image()    
    #DARK IMAGE MAGIC
    img_normalizied_1 = img_normalizied#img_normalized
    image_shape = img_normalizied.shape
    img_normalizied = img_normalizied.reshape(image_shape[0],-1)
    image_shape = img_normalizied.shape
    inner_fold = model.layers[0].in_fold
    tmp_in = img_normalizied.reshape(image_shape[0], inner_fold, int(image_shape[1]/inner_fold))
    tmp_in = tmp_in[:,:,model.in_perm.long()]

    layer_iterator = model.layer_Output_Generator(img_normalizied_1)
    for luLayer, weights, layer_number in layer_iterator:
        # BLACK TENSOR MAGIC
        tmp_out = luLayer
        tmp_tensor_out = torch.squeeze(tmp_out).unsqueeze(1)
        tmp_tensor_in = torch.squeeze(tmp_in)
        tmp_in = tmp_out
        transition = tmp_tensor_out * weights
        max_path_value= transition.max()
        relevant_paths = (transition >= max_path_value/1.5).nonzero(as_tuple=False)
        for g in relevant_paths:
            i = g[1]
            j = g[0]
            x = i * 100/tmp_tensor_in.shape[0]
            y = j * 100/tmp_tensor_out.shape[0]
            alpha = (transition[j,i]/max_path_value).item() if transition[j,i]/max_path_value>0.35 else 0
            image = draw_connection_in_image(image,layer_number,x,y,CMAP(alpha))

        if SHOWFIGURE:
            #TODO
            #Send Image To Wall
            pass
    if SAVEFIGURE:
        image.save('./results/mnist/{0:06d}.png'.format(step-1),"PNG")
    return None

## MAIN TRAINING LOOP
def yield_training_loop():
    train = load_Data("train")
    test = load_Data("test")
    val = load_Data("val")
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

    CLASSES = get_classes(DATADIR)

    # Shape = layersize
    model = BioMLP2D(shape=(784,100,100,100,100,100, 10))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3, weight_decay=0.0)

    one_hots = torch.eye(10,10).to(DEVICE)

    model.eval()
    #print("init accuraxy: {0:.4f}".format(accuracy(model,test,device=DEVICE)))

    test_accuracies = []
    train_accuracies = []

    step = 0
    model.train()

    best_train_loss = 1e4#1e4
    best_test_loss = 1e4#1e4
    best_train_accuracies = 0
    best_test_accuracies = 0

    log = 200#200
    lamb = 0.0005#0.01
    weight_factor = 1.0#2.0
    swap_log = 50#500
    plot_log = 25#200#250
    rand_log = 5

    pbar = tqdm(islice(cycle(train_loader), STEPS), total=STEPS,disable=True)

    for x, label in pbar:
        if step == int(step/4):
            lamb *= 10
        if step == int(step/2):
            lamb *= 10
        
        model.train()
        optimizer.zero_grad()
        loss_train = loss_function(model(x.to(DEVICE)), one_hots[label.squeeze().long()])
        cc = model.get_compute_connection_cost(weight_factor=weight_factor,no_penalize_last=True)
        total_loss = loss_train + lamb*cc
        total_loss.backward()
        optimizer.step()

        if step % log == 0:
            with torch.no_grad():
                model.eval()
                train_accuracies = accuracy(model, train, device=DEVICE).item()
                test_accuracies = accuracy(model, test, device=DEVICE).item()
                train_loss = loss_Function(model,train,DEVICE).item()
                test_loss = loss_Function(model, test, DEVICE).item()

                model.train()
                pbar.set_description("{:3.3f} | {:3.3f} | {:3.3f} | {:3.3f} | {:3.3f} ".format(train_accuracies, test_accuracies, train_loss, test_loss, cc))
        if step % rand_log == 0:
            #TODO
            #Show some Random images
            pass 
        if step % swap_log  == 0:
            model.relocate()
        if (step -1) % plot_log == 0:
            print('Awaiting Input')
            yield model, CLASSES#Image Classification
        step += 1

if __name__ == '__main__':
    pass