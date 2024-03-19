import numpy
import torch
import torchvision
import matplotlib.pyplot as pyplot

from PIL import Image
from tqdm.auto import tqdm
from itertools import islice
from netzmodelle import BioLinear2D
from netzmodelle import BioMLP2D


SEED = 1
numpy.random.seed(SEED)
torch.manual_seed(SEED)

TEST_IMAGE_PATH = "/home/thappek/Documents/data/MNIST/sample_image.webp"

BATCHSIZE = 50
SHUFFLE =True

TRAIN = torchvision.datasets.MNIST(root="/tmp", train=True, transform=torchvision.transforms.ToTensor(), download=True)
TEST = torchvision.datasets.MNIST(root="/tmp", train=False, transform=torchvision.transforms.ToTensor(), download=True)
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN, BATCHSIZE, SHUFFLE)

DATASIZE = 60000
STEPS = 40000

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def accuracy(network, dataset, device, N=2000, batch_size=BATCHSIZE):
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    correct = 0
    total = 0
    for x, labels in islice(dataset_loader, N // batch_size):
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
        logits = network(x.to(device))
        loss += torch.sum((logits-torch.eye(10,)[labels])**2)
        total += x.size(0)
    return loss/total
    
def eval_image(model, image_path):
    image = Image.open(image_path).convert('L')
    mean = [0.5] 
    std = [0.5]
    transform_norm = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), 
    torchvision.transforms.Resize((28,28)),torchvision.transforms.Normalize(mean, std)])
    img_normalized = transform_norm(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(DEVICE)
    with torch.no_grad():
        model.eval()  
        logits = mlp(img_normalized.to(DEVICE))
        predicted_labels = torch.argmax(logits, dim=1)
        
        return predicted_labels[0]
    
def render_image(model):
    # ADD FIGURE
    number_of_steps = 100
    figure = pyplot.figure(num=1, clear=True)
    axes = figure.add_subplot()
    number_of_layers = len(model.layers)
    # PLOT NODES
    for layer_number in range(1,number_of_layers):
        biolayer = model.layers[layer_number]
        number_of_nodes = biolayer.linear.weight.shape[1]
        nodes_distribution = numpy.full((number_of_nodes),range(number_of_nodes))
        height = numpy.full(model.layers[layer_number].in_coordinates[:,0].detach().numpy().shape,layer_number)
        axes.scatter(nodes_distribution,height,s=5,alpha=0.5,color="black")
    #    pass
    axes.scatter(numpy.full((784),numpy.linspace(0,100,784,endpoint=False)),numpy.full(model.layers[0].in_coordinates[:,0].detach().numpy().shape,1),s=5,alpha=0.5,color="black")
    #axes.scatter(numpy.full((100),range(100)),numpy.full(model.layers[1].in_coordinates[:,0].detach().numpy().shape,1),s=5,alpha=0.5,color="black")
    #axes.scatter(numpy.full((100),range(100)),numpy.full(model.layers[2].in_coordinates[:,0].detach().numpy().shape,2),s=5,alpha=0.5,color="black")
    axes.scatter(numpy.full((10),range(0,100,10)),numpy.full(model.layers[-1].out_coordinates[:,0].detach().numpy().shape,2),s=5,alpha=0.5,color="black")
    # GET THE WEIGHTS
    #out_layer = mlp.layers[-1].out_coordinates.detach().numpy()

    
    #for layer_number in range(number_of_layers-1):
    for layer_number in range(0,3):
        biolayer = model.layers[layer_number]
        bio_weights = biolayer.linear.weight.clone()
        bio_weights_shape = bio_weights.shape
        bio_weights = bio_weights/torch.abs(bio_weights).max()
        lw = 0
        for i in range(bio_weights_shape[0]):
            for j in range(bio_weights_shape[1]):
                #out_weights = biolayer.out_coordinates[i].detach().numpy()
                #in_weights = biolayer.in_coordinates[i].detach().numpy()
                x = i * 100/bio_weights_shape[0]
                y = j * 100/bio_weights_shape[1]
                pyplot.plot([x,y], [layer_number+1,layer_number], lw=2*numpy.abs(bio_weights[i,j].detach().numpy()), color="blue" if bio_weights[i,j]>0 else "red")
    #pyplot.show()
    pyplot.savefig('./results/mnist/{0:06d}.png'.format(step-1))


if __name__ == '__main__':
    train = torch.utils.data.Subset(TRAIN, range(DATASIZE))
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

    width = 200
    # Shape = layersize
    mlp = BioMLP2D(shape=(784,100,100,10))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(),lr=1e-3, weight_decay=0.0)

    one_hots = torch.eye(10,10).to(DEVICE)

    mlp.eval()
    print("init accuraxy: {0:.4f}".format(accuracy(mlp,TEST,device=DEVICE)))

    test_accuracies = []
    train_accuracies = []

    step = 0
    mlp.train()

    best_train_loss = 1e4
    best_test_loss = 1e4
    best_train_accuracies = 0
    best_test_accuracies = 0

    log = 200
    lamb = 0.01
    swap_log = 500
    plot_log = 100

    pbar = tqdm(islice(cycle(train_loader), STEPS), total=STEPS)

    for x, label in pbar:
        if step == int(step/4):
            lamb *= 10
        if step == int(step/2):
            lamb *= 10
        mlp.train()
        optimizer.zero_grad()
        loss_train = loss_function(mlp(x.to(DEVICE)), one_hots[label])
        cc = mlp.get_compute_connection_cost(weight_factor=2.0,no_penalize_last=True)
        total_loss = loss_train + lamb*cc
        total_loss.backward()
        optimizer.step()

        if step % log == 0:
            with torch.no_grad():
                mlp.eval()
                train_accuracies = accuracy(mlp, train, device=DEVICE).item()
                test_accuracies = accuracy(mlp, TEST, device=DEVICE).item()
                train_loss = loss_Function(mlp,train,DEVICE).item()
                test_loss = loss_Function(mlp, TEST, DEVICE).item()

                mlp.train()
                pbar.set_description("{:3.3f} | {:3.3f} | {:3.3f} | {:3.3f} | {:3.3f} ".format(train_accuracies, test_accuracies, train_loss, test_loss, cc))
        step += 1

        if step % swap_log  == 0:
            mlp.relocate()
        if (step -1) % plot_log == 0:
            #Image Classification
            image_class = eval_image(mlp, TEST_IMAGE_PATH)
            print(image_class)
            render_image(mlp)
            print("trap")
