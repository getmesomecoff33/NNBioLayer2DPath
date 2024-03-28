import numpy
import torch
import torchvision
import matplotlib.pyplot as pyplot

from PIL import Image
from tqdm.auto import tqdm
from itertools import islice
from netzmodelle import BioMLP2D

SHOWFIGURE = False
SAVEFIGURE = True

DEBUG = True

SEED = 1
numpy.random.seed(SEED)
torch.manual_seed(SEED)

TEST_IMAGE_PATH = "/home/thappek/Documents/data/MNIST/32.png"

BATCHSIZE = 50
SHUFFLE = True

TRAIN = torchvision.datasets.MNIST(root="/tmp", train=True, transform=torchvision.transforms.ToTensor(), download=True)
TEST = torchvision.datasets.MNIST(root="/tmp", train=False, transform=torchvision.transforms.ToTensor(), download=True)
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN, BATCHSIZE, SHUFFLE)

DATASIZE = 60000
STEPS = 40000

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def color_Map(value):
    if value > 0.9:
        return "lightcoral"
    if value > 0.7:
        return 'teal'
    if value > 0.5:
        return 'cadetblue'
    if value > 0.3:
        return "cadetblue"
    return 'darkslategray'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def accuracy(network, dataset, device, N=2000, batch_size=BATCHSIZE):
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    total = 0
    correct = 0
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

def load_image(image_path):
    std = [0.5]
    mean = [0.5]
    image = Image.open(image_path).convert("L")
    transform_norm = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), 
    torchvision.transforms.Resize((28,28)),torchvision.transforms.Normalize(mean, std)])
    img_normalized = transform_norm(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(DEVICE)
    return img_normalized
    
def eval_image(model, image_path):
    img_normalized = load_image(image_path=image_path)
    with torch.no_grad():
        model.eval()  
        logits = model(img_normalized.to(DEVICE))
        predicted_labels = torch.argmax(logits, dim=1)
        if DEBUG:
            return logits
        return predicted_labels[0]

def render_image(model,image_path=None):
    render_image_path(model=model,image_path=image_path)

def render_image_path(model,image_path):
    #TODO
    #Tensor label scheint nicht mit der scatternode uebereinzustimmen
    # Bug in letzetn layer painter
    # ADD FIGURE
    number_of_steps = 100
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
    
    img_normalizied_1 = load_image(image_path=image_path)
    img_normalizied = img_normalizied_1
    #DARK IMAGE MAGIC
    image_shape = img_normalizied.shape
    img_normalizied = img_normalizied.reshape(image_shape[0],-1)
    image_shape = img_normalizied.shape
    inner_fold = model.layers[0].in_fold
    tmp_in = img_normalizied.reshape(image_shape[0], inner_fold, int(image_shape[1]/inner_fold))
    tmp_in = tmp_in[:,:,model.in_perm.long()]

    layer_iterator = model.layer_Output_Generator(img_normalizied_1)
    for tmp_out, weights, layer_number in layer_iterator:
        # BLACK TENSOR MAGIC
        tmp_tensor_out = torch.squeeze(tmp_out).unsqueeze(1)
        tmp_tensor_in = torch.squeeze(tmp_in)
        tmp_in = tmp_out
        transition = tmp_tensor_in * weights
        if layer_number == number_of_layers-1:    
            print("")
        max_path_value= transition.max()
        
        for i in range(tmp_tensor_in.shape[0]):
            for j in range(tmp_tensor_out.shape[0]):
                x = i * 100/tmp_tensor_in.shape[0]
                y = j * 100/tmp_tensor_out.shape[0]
                alpha = (transition[j,i]/max_path_value).item() if transition[j,i]/max_path_value>0.25 else 0
                pyplot.plot([x,y], [layer_number,layer_number+1], lw=1, alpha=alpha, color=color_Map(alpha))

    if SHOWFIGURE:
        pyplot.show()
    if SAVEFIGURE:
        pyplot.savefig('./results/mnist/{0:06d}.png'.format(step-1))


if __name__ == '__main__':
    train = torch.utils.data.Subset(TRAIN, range(DATASIZE))
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

    width = 200
    # Shape = layersize
    mlp = BioMLP2D(shape=(784,100,100,100,10))
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
    swap_log = 50#500
    plot_log = 50#250

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
            render_image(mlp,TEST_IMAGE_PATH)
