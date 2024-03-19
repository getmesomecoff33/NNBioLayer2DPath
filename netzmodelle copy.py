import numpy
import torch

SEED = 1
numpy.random.seed(SEED)
torch.manual_seed(SEED)

class Bio2D(torch.nn.Module):
    inner_coordinates = None
    outer_coordinates = None
    inner_dimension = None
    outer_dimension = None
    inner_fold = None
    outer_fold = None
    #linear = None

    def __init__(self, inner_dimension, outer_dimension, inner_fold=1, outer_fold=1, outer_ring=False):
        super(Bio2D,self).__init__()
        assert inner_dimension % inner_fold == 0
        assert outer_dimension % outer_fold == 0

        self.inner_dimension = inner_dimension
        self.outer_dimension = outer_dimension
        self.inner_fold = inner_fold
        self.outer_fold = outer_fold
        self.linear = torch.nn.Linear(inner_dimension, outer_dimension)

        inner_dimension_fold = int(inner_dimension/inner_fold)
        outer_dimension_fold = int(outer_dimension/outer_fold)
        inner_dimension_fold_square = int(numpy.sqrt(inner_dimension_fold))
        outer_dimension_fold_square = int(numpy.sqrt(outer_dimension_fold))

        start = 1/(2*inner_dimension_fold_square)
        stop = 1-start
        inner_vector = numpy.linspace(start,stop,num=inner_dimension_fold_square)
        X,Y = numpy.meshgrid(inner_vector,inner_vector)
        self.inner_coordinates = torch.tensor(numpy.transpose(numpy.array([X.reshape(-1,),Y.reshape(-1,)])),dtype=torch.float)

        start = 1/(2*outer_dimension_fold_square)
        stop = 1-start
        outer_vector = numpy.linspace(start,stop, num=outer_dimension_fold_square)
        X,Y = numpy.meshgrid(outer_vector,outer_vector)
        self.outer_coordinates = torch.tensor(numpy.transpose(numpy.array([X.reshape(-1,),Y.reshape(-1,)])),dtype=torch.float)
        if outer_ring:
            alpha = 1/(2*outer_dimension_fold)*2*numpy.pi
            thetas = numpy.linspace(alpha,1-alpha,num=outer_dimension_fold)
            self.outer_coordinates = 0.5*torch.tensor(numpy.transpose(numpy.array([numpy.cos(thetas),numpy.sin(thetas)]))/4,dtype=torch.float) 

    def forward(self,x):
        return self.linear(x)

## NETZ MODEL
class BioModel(torch.nn.Module):
    distance_between_layers = 0.5
    top_k = 30
    
    inner_dimension = None
    outer_dimension = None
    token_embedding = None
    #inner_parameter = None
    #outer_parameter = None
    embedding = None
    #linears = None
    shape = None
    depth = None
    
    def __init__(self, inner_dimension=2, outer_dimension=2, w=2, depth=2, shape=None,token_embedding=False, embedding_size=None):
        super(BioModel,self).__init__()
        linear_list = []
        self.shape = [inner_dimension] + [w]*(depth-1) + [outer_dimension]
        self.inner_dimension = inner_dimension
        self.outer_dimension = outer_dimension
        self.depth = depth

        if shape:
            self.shape = shape
            self.inner_dimension = shape[0]
            self.outer_dimension = shape[-1]
            self.depth  = len(shape)-1
        
        for i in range(depth):
            if i == 0:
                linear_list.append(Bio2D(self.shape[i],self.shape[i+1], inner_fold=1))
                continue
            if i == depth-1:
                linear_list.append(Bio2D(self.shape[i],self.shape[i+1], inner_fold=1,outer_ring=True))
                continue
            linear_list.append(Bio2D(self.shape[i],self.shape[i+1]))
        self.linears = torch.nn.ModuleList(linear_list)

        if token_embedding:
            self.embedding = torch.nn.Parameter(torch.normal(0,1,size=embedding_size))
        self.inner_parameter = torch.nn.Parameter(torch.tensor(numpy.arange(int(self.inner_dimension/self.linears[0].inner_fold)), dtype=torch.float))
        self.outer_parameter = torch.nn.Parameter(torch.tensor(numpy.arange(int(self.outer_dimension/self.linears[-1].outer_fold)), dtype=torch.float))
        

        
    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0],-1)
        shape = x.shape
        inner_fold = self.linears[0].inner_fold
        x = x.reshape(shape[0],inner_fold, int(shape[-1]/inner_fold))
        x = x[:,:,self.inner_parameter.long()]
        x = x.reshape(shape[0],shape[1])
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x=f(self.linears[i](x))
        x = self.linears[-1](x)
        outer_parameter_inverse = torch.zeros(self.outer_dimension,dtype=torch.long)
        outer_parameter_inverse[self.outer_parameter.long()] = torch.arange(self.outer_dimension)
        return x

    def get_Linear_Layers(self):
        return self.linears
    
    def get_CC(self, weight_factor=0.2, bias_penalize=True, no_penalize_last=False):
        cc = 0
        number_of_linears = len(self.linears)
        for i in range(number_of_linears):
            if i == number_of_linears-1 and no_penalize_last:
                weight_factor = 0
            biolinear = self.linears[i]
            dist = torch.sum(torch.abs(biolinear.outer_coordinates.unsqueeze(dim=1) - biolinear.innner_coordinates.unsqueeze(dim=0)),dim=2)
            cc += torch.mean(torch.abs(biolinear.linear.weight)*(weight_factor*dist+self.l0))
            if bias_penalize:
                cc += torch.mean(torch.abs(biolinear.linear.bias)*(self.l0))
        if self.token_embedding:
            cc += torch.mean(torch.abs(self.embedding)*(self.l0))
        return cc


    def swap_Weights(self, weights, j, k, swap_type='out'):
        with torch.no_grad():
            if swap_type == 'in':
                tmp = weights[:,j].clone()
                weights[:,j] = weights[:,k].clone()
                weights[:,k] = tmp
                return True
            if swap_type == 'out':
                tmp = weights[j].clone()
                weights[j] = weights[k].clone
                weights[k] = tmp
                return True
        raise Exception('Something important broke and we are all fucked')

    def swap_Bias(self, biases, j, k):
        with torch.no_grad():
            tmp = biases[j].clone()
            biases[j] = biases[k].clone()
            biases[k] = tmp
        return True

    def swap(self, i, j, k):
        #Swap neurons j and k in the ith layer
        layers = self.get_Linear_Layers()
        number_of_layers = len(layers)
        if i == 0:
            return True
        if i == number_of_layers:
            weights = layers[i-1].linear.weight
            biases = layers[i-1].linear.bias
            self.swap_Weights(weights=weights, j=j, k=k , swap_type='out')
            self.swap_Bias(biases=biases, j=j, k=k,)
            self.swap_Bias(self.outer_parameter,j=j,k=k)
            return True
        weights_in = layers[i-1].linear.weight
        weights_out = layers[i].linear.weight
        biases = layers[i-1].linear.bias
        self.swap_Weights(weights=weights_in,j=j,k=k, swap_type='out')
        self.swap_Weights(weights=weights_out,j=j,k=k,swap_type='in')
        self.swap_Bias(biases=biases,j=j,k=k)
        return True


    def get_Top_ID(self, i, top_k=20):
        layers = self.get_Linear_Layers()
        number_of_layers = len(layers)
        if i == 0:
            weights = layers[i].linear.weight
            score = torch.sum(torch.abs(weights),dim=1)
            inner_flod = layers[0].inner_fold
            score =  torch.sum(score.reshape(inner_flod, int(score.shape[0]/inner_flod)), dim=1)
            top_index = torch.flip(torch.argsort(score),[0])[:top_k]
            return top_index, score
        if i == number_of_layers:
            weights = layers[i-1].linear.weight
            score = torch.sum(torch.abs(weights),dim=1)
            top_index = torch.flip(torch.argsort(score),[0])[:top_k]
            return top_index, score
        weights_in = layers[i-1].linear.weight
        weights_out = layers[i].linear.weight
        score = torch.sum(torch.abs(weights_in),dim=1) + torch.sum(torch.abs(weights_out),dim=0)
        top_index = torch.flip(torch.argsort(score),[0])[:top_k]
        return top_index, score       

    def relocate_ji(self, i, j):
        #realocate the jth neuron in layer i
        ccs = []
        layers =self.get_Linear_Layers()
        number_of_layers = len(layers)
        if i < number_of_layers:
            number_of_neuron = int(layers[i].linear.weight.shape[1]/layers[i].inner_fold)
        if i >= number_of_layers:
            number_of_neuron = int(layers[i-1].linear.weight.shape[0])        
        for k in range(number_of_neuron):
            self.swap(i,j,k)
            ccs.append(self.get_CC)
            self.swap(i,j,k)
        k = torch.argmin(torch.stack(ccs))
        self.swap(i,j,k)

    def relocate_i(self, i):
        top_id = self.get_Top_ID(i, top_k=self.top_k)
        for j in top_id[0]:
            self.relocate_ji(i,j)

    def relocate(self):
        layers = self.get_Linear_Layers()
        number_of_layers = len(layers)
        for i in range(number_of_layers+1):
            self.relocate_i(i)

## ENDE NETZ MODEL


if __name__ == '__main__':
    pass