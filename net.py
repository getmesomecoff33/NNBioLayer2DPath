import os
import json
import numpy
import torch
import random
import torch.utils
import torch.utils.data
import torchvision
import matplotlib.pyplot as pyplot

from PIL import Image
from tqdm.auto import tqdm
from itertools import islice
from netzmodelle import BioMLP2D

SHOWFIGURE = False
SAVEFIGURE = True

DEBUG = False
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

TEST_IMAGE_PATH = IMAGE_DIR + IMAGE_NAME
DATADIR = os.path.join(DATAPATH,RAWDATA)

BATCHSIZE = 50
SHUFFLE = True


IMAGETRANSFORMER = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28,28))])

DATASIZE = 60000
STEPS = 40000

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#Color Coding
CMAP = pyplot.colormaps.get_cmap('jet')
LINEWIDTH = 1

CLASSES = []

def load_Data(subset):
    #TODO load the created trainingsdataset
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

def invert_image(image):
    invertedImage = 1-image
    imgWhithoutNoise = torch.nn.functional.relu(invertedImage-0.01)
    imgWhithoutNoise[imgWhithoutNoise!=0] = 1
    return imgWhithoutNoise

def load_image_from_path(image_path):
    image = Image.open(image_path).convert("L")    
    img_normalized = IMAGETRANSFORMER(image).float()
    img_inverted = invert_image(img_normalized)
    #img_normalized = img_normalized.to(DEVICE)
    return img_inverted

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

def render_image_path(model,img_normalizied):
    #TODO
    #Tensor label scheint nicht mit der scatternode uebereinzustimmen
    # Bug in letzetn layer painter
    # ADD FIGURE
    figure = pyplot.figure(num=1, clear=True)
    axes = figure.add_subplot()
    number_of_layers = len(model.layers)
    # PLOT NODES
    # First Layer
    axes.scatter(numpy.full((model.layers[0].linear.weight.shape[0]),numpy.linspace(0,100,model.layers[0].linear.weight.shape[0],endpoint=False)),numpy.full(model.layers[0].out_coordinates[:,0].detach().numpy().shape,0),s=5,alpha=0.5,color="black")
    # Middle Layer
    for layer_number in range(1,number_of_layers):
        biolayer = model.layers[layer_number]
        number_of_nodes = biolayer.linear.weight.shape[1]
        nodes_distribution = numpy.full((number_of_nodes),range(number_of_nodes))
        height = numpy.full(model.layers[layer_number].in_coordinates[:,0].detach().numpy().shape,layer_number)
        axes.scatter(nodes_distribution,height,s=5,alpha=0.5,color="black")
    #Output Layer
    axes.scatter(numpy.full((model.layers[-1].linear.weight.shape[0]),range(0,100,model.layers[-1].linear.weight.shape[0])),numpy.full(model.layers[-1].out_coordinates[:,0].detach().numpy().shape,number_of_layers),s=5,alpha=0.5,color="black")
    
    
    
    #DARK IMAGE MAGIC
    img_normalizied_1 = img_normalized
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
        #if layer_number == number_of_layers - 1:
         #   tmp_out = purLayer
        tmp_tensor_out = torch.squeeze(tmp_out).unsqueeze(1)
        tmp_tensor_in = torch.squeeze(tmp_in)
        tmp_in = tmp_out
        #transition = tmp_tensor_in * weights
        transition = tmp_tensor_out * weights
        if layer_number == number_of_layers-1:    
            print("")
        max_path_value= transition.max()
        
        for i in range(tmp_tensor_in.shape[0]):
            for j in range(tmp_tensor_out.shape[0]):
                x = i * 100/tmp_tensor_in.shape[0]
                y = j * 100/tmp_tensor_out.shape[0]
                alpha = (transition[j,i]/max_path_value).item() if transition[j,i]/max_path_value>0.35 else 0
                pyplot.plot([x,y], [layer_number,layer_number+1], lw=LINEWIDTH, alpha=alpha, color=CMAP(alpha))

    if SHOWFIGURE:
        pyplot.show()
    if SAVEFIGURE:
        pyplot.savefig('./results/mnist/{0:06d}.png'.format(step-1))


if __name__ == '__main__':
    #train = torch.utils.data.Subset(TRAIN, range(DATASIZE))
    train = load_Data("train")
    test = load_Data("test")
    val = load_Data("val")
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

    CLASSES = get_classes(DATADIR)

    # Shape = layersize
    model = BioMLP2D(shape=(784,100,100,100,100,10))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3, weight_decay=0.0)

    one_hots = torch.eye(10,10).to(DEVICE)

    model.eval()
    print("init accuraxy: {0:.4f}".format(accuracy(model,test,device=DEVICE)))

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
    weight_factor = 2.0#2.0
    swap_log = 50#500
    plot_log = 200#250

    pbar = tqdm(islice(cycle(train_loader), STEPS), total=STEPS)

    for x, label in pbar:
        if step == int(step/4):
            lamb *= 10
        if step == int(step/2):
            lamb *= 10
        model.train()
        optimizer.zero_grad()
        #loss_train = loss_function(model(x.to(DEVICE)), one_hots[label.long()])
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

        if step % swap_log  == 0:
            model.relocate()
        if (step -1) % plot_log == 0:
            #Image Classification
            if LOADFROMDATASET:
                img_normalized, label = load_image_from_dataset(val)
            if not LOADFROMDATASET:
                img_normalized = load_image_from_path(image_path=TEST_IMAGE_PATH)
            predictedLabel, certainty = eval_image(model, img_normalized=img_normalized)
            imageClass = CLASSES[predictedLabel]
            print("I thinke it's a {} with {} percent certainty".format(imageClass, certainty*100))
            if LOADFROMDATASET:
                if not predictedLabel == label:
                    realLabel = CLASSES[label]
                    print("In reality its a {} you piece of...".format(realLabel))
                if predictedLabel == label:
                    print("Good job, just lucky though...")

            render_image_path(model,img_normalized)
            pass
        step += 1