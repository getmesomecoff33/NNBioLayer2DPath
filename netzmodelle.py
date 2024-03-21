import numpy as np
import torch.nn as nn
import torch

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

class BioLinear2D(nn.Module):

    def __init__(self, in_dim, out_dim, in_fold=1, out_fold=1, out_ring=False):
        super(BioLinear2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_fold = in_fold
        self.out_fold = out_fold
        assert in_dim % in_fold == 0
        assert out_dim % out_fold == 0
        in_dim_fold = int(in_dim/in_fold)
        out_dim_fold = int(out_dim/out_fold)
        in_dim_sqrt = int(np.sqrt(in_dim_fold))
        out_dim_sqrt = int(np.sqrt(out_dim_fold))
        x = np.linspace(1/(2*in_dim_sqrt), 1-1/(2*in_dim_sqrt), num=in_dim_sqrt)
        X, Y = np.meshgrid(x, x)
        self.in_coordinates = torch.tensor(np.transpose(np.array([X.reshape(-1,), Y.reshape(-1,)])), dtype=torch.float)
        if out_ring:
            thetas = np.linspace(1/(2*out_dim_fold)*2*np.pi, (1-1/(2*out_dim_fold))*2*np.pi, num=out_dim_fold)
            self.out_coordinates = 0.5+torch.tensor(np.transpose(np.array([np.cos(thetas), np.sin(thetas)]))/4, dtype=torch.float)
        else:
            x = np.linspace(1/(2*out_dim_sqrt), 1-1/(2*out_dim_sqrt), num=out_dim_sqrt)
            X, Y = np.meshgrid(x, x)
            self.out_coordinates = torch.tensor(np.transpose(np.array([X.reshape(-1,), Y.reshape(-1,)])), dtype=torch.float)

        
    def forward(self, x):
        return self.linear(x)

## NETZ MODEL
class BioMLP2D(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, w=2, depth=2, shape=None, token_embedding=False, embedding_size=None):
        super(BioMLP2D, self).__init__()
        if shape == None:
            shape = [in_dim] + [w]*(depth-1) + [out_dim]
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.depth = depth
        else:
            self.in_dim = shape[0]
            self.out_dim = shape[-1]
            self.depth = len(shape) - 1
        layer_list = []
        for i in range(self.depth):
            if i == 0:
                layer_list.append(BioLinear2D(shape[i], shape[i+1], in_fold=1))
            elif i == self.depth - 1:
                layer_list.append(BioLinear2D(shape[i], shape[i+1], in_fold=1, out_ring=True))
            else:
                layer_list.append(BioLinear2D(shape[i], shape[i+1]))
        self.layers = nn.ModuleList(layer_list)  
        if token_embedding == True:
            self.embedding = torch.nn.Parameter(torch.normal(0,1,size=embedding_size))
        self.shape = shape
        self.l0 = 0.5 
        self.in_perm = nn.Parameter(torch.tensor(np.arange(int(self.in_dim/self.layers[0].in_fold)), dtype=torch.float))
        self.out_perm = nn.Parameter(torch.tensor(np.arange(int(self.out_dim/self.layers[-1].out_fold)), dtype=torch.float))
        self.top_k = 30
        self.token_embedding = token_embedding

    def forward(self, x):
        shp = x.shape
        x = x.reshape(shp[0],-1)
        shp = x.shape
        in_fold = self.layers[0].in_fold
        x = x.reshape(shp[0], in_fold, int(shp[1]/in_fold))
        x = x[:,:,self.in_perm.long()]
        x = x.reshape(shp[0], shp[1])
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = f(self.layers[i](x))
        x = self.layers[-1](x)
        
        out_perm_inv = torch.zeros(self.out_dim, dtype=torch.long)
        out_perm_inv[self.out_perm.long()] = torch.arange(self.out_dim)
        x = x[:,out_perm_inv]
        return x
    
    def eval_layer(self, x, layer_number):
        shp = x.shape
        x = x.reshape(shp[0],-1)
        shp = x.shape
        in_fold = self.layers[0].in_fold
        x = x.reshape(shp[0], in_fold, int(shp[1]/in_fold))
        x = x[:,:,self.in_perm.long()]
        x = x.reshape(shp[0], shp[1])
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = f(self.layers[i](x))
            if layer_number == i:
                return x
        x = self.layers[-1](x)
        return x
    
    def get_linear_layers(self):
        return self.layers
    
    def get_compute_connection_cost(self, weight_factor=2.0, bias_penalize=True, no_penalize_last=False):
        ccc = 0
        number_of_layers = len(self.layers)
        for i in range(number_of_layers):
            if i == number_of_layers - 1 and no_penalize_last:
                weight_factor = 0.
            biolinear = self.layers[i]
            dist = torch.sum(torch.abs(biolinear.out_coordinates.unsqueeze(dim=1) - biolinear.in_coordinates.unsqueeze(dim=0)),dim=2)
            ccc += torch.mean(torch.abs(biolinear.linear.weight)*(weight_factor*dist+self.l0))
            if bias_penalize == True:
                ccc += torch.mean(torch.abs(biolinear.linear.bias)*(self.l0))
        if self.token_embedding:
            ccc += torch.mean(torch.abs(self.embedding)*(self.l0))
        return ccc
    
    def swap_weight(self, weights, j, k, swap_type="out"):
        with torch.no_grad():  
            if swap_type == "in":
                temp = weights[:,j].clone()
                weights[:,j] = weights[:,k].clone()
                weights[:,k] = temp
                return
            if swap_type == "out":
                temp = weights[j].clone()
                weights[j] = weights[k].clone()
                weights[k] = temp
                return
            raise Exception("Swap type {} is not recognized!".format(swap_type))
            
    def swap_bias(self, biases, j, k):
        with torch.no_grad():  
            temp = biases[j].clone()
            biases[j] = biases[k].clone()
            biases[k] = temp
    
    def swap(self, i, j, k):
        # in the ith layer (of neurons), swap the jth and the kth neuron. 
        # Note: n layers of weights means n+1 layers of neurons.
        # (incoming, outgoing) * weights + biases are swapped. 
        layers = self.get_linear_layers()
        number_of_layers = len(layers)
        if i == 0:
            return
        if i == number_of_layers:
            weights = layers[i-1].linear.weight
            biases = layers[i-1].linear.bias
            self.swap_weight(weights, j, k, swap_type="out")
            self.swap_bias(biases, j, k)
            self.swap_bias(self.out_perm, j, k)
            return
        weights_in = layers[i-1].linear.weight
        weights_out = layers[i].linear.weight
        biases = layers[i-1].linear.bias
        self.swap_weight(weights_in, j, k, swap_type="out")
        self.swap_weight(weights_out, j, k, swap_type="in")
        self.swap_bias(biases, j, k)

    def get_top_id(self, i, top_k=20):
        layers = self.get_linear_layers()
        number_of_linear = len(layers)
        if i == 0:
            weights = layers[i].linear.weight
            score = torch.sum(torch.abs(weights), dim=0)
            in_fold = layers[0].in_fold
            score = torch.sum(score.reshape(in_fold, int(score.shape[0]/in_fold)), dim=0)
            top_index = torch.flip(torch.argsort(score),[0])[:top_k]
            return top_index, score
        if i == number_of_linear:
            weights = layers[i-1].linear.weight
            score = torch.sum(torch.abs(weights), dim=1)
            top_index = torch.flip(torch.argsort(score),[0])[:top_k]
            return top_index, score
        weights_in = layers[i-1].linear.weight
        weights_out = layers[i].linear.weight
        score = torch.sum(torch.abs(weights_out), dim=0) + torch.sum(torch.abs(weights_in), dim=1)
        top_index = torch.flip(torch.argsort(score),[0])[:top_k]
        return top_index, score
    
    def relocate_neuron_ij(self, i, j):
        layers = self.get_linear_layers()
        number_of_layers = len(layers)
        if i < number_of_layers:
            number_of_neurons = int(layers[i].linear.weight.shape[1]/layers[i].in_fold)
        else:
            number_of_neurons = layers[i-1].linear.weight.shape[0]
        ccs = []
        for k in range(number_of_neurons):
            self.swap(i,j,k)
            ccs.append(self.get_compute_connection_cost())
            self.swap(i,j,k)
        k = torch.argmin(torch.stack(ccs))
        self.swap(i,j,k)
            
    def relocate_neurons_in_layer(self, i):
        top_id = self.get_top_id(i, top_k=self.top_k)
        for j in top_id[0]:
            self.relocate_neuron_ij(i,j)
            
    def relocate(self):
        layers = self.get_linear_layers()
        number_layers = len(layers)
        for layer_number in range(number_layers+1):
            self.relocate_neurons_in_layer(layer_number)
            
    

## ENDE NETZ MODEL


if __name__ == '__main__':
    pass