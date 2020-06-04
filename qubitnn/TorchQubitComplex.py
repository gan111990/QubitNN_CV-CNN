#!/home/kumar/anaconda3/envs/fastai-cuda/bin/python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
# from ignite.metrics import Metric, Accuracy
from matplotlib import pyplot as plt


# In[2]:


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev


# In[3]:


def sigmoid(X):
    return 1 / (1 + torch.exp(-X))

def gaussian(X,a=.0,b=1.0):
    return torch.exp(-((X-a)/b)**2)

def qubitPhaseLimitLayer(body):
    #x_max = mx.sym.max(body, axis = 0)
    x_max = torch.max(body)
    #normalized = mx.sym.transpose(mx.sym.transpose(body) / x_max)
    normalized = torch.transpose(torch.transpose(body,0,1) / x_max, 0, 1)
    half_pi_normalized = normalized * np.pi / 2

    return half_pi_normalized

def get_re_im_from_fi(fi,coeff = 1):
    re = coeff * torch.cos(fi)
    im = coeff * torch.sin(fi)
    return re, im


# In[4]:


class QubitComplex(torch.nn.Module):
    def __init__( self, in_shape=None, out_units=None, magnitude=.1, no_bias=False, out_layer=False, power = 2,
                 variant = 1, out_trainable=False, layer_num=1, complex_input=False,
                 device=torch.device('cpu') ):
        super( QubitComplex, self).__init__()
        self.eps = 0.001 
        self.layer_num = layer_num
        self.shape1 = None
        self.shape2 = None
        self.no_bias = no_bias
        self.out_layer = out_layer
        self.variant   = variant
        self.power     = power
        self.out_trainable = out_trainable
        self.magnitude = magnitude
        self.device = device
        if in_shape:
            self.shape1 = in_shape
        if out_units:
            if isinstance(out_units, tuple):
                self.shape2 = out_units
            else:
                self.shape2 = (out_units,)
        #print(self.shape1,self.shape2)
        theta_re = torch.empty( (self.shape1), requires_grad=True, device=device)
        self.theta_re = torch.nn.Parameter(torch.nn.init.xavier_uniform_(theta_re, gain=self.magnitude))
        theta_im = torch.empty( (self.shape1), requires_grad=True, device=device)
        self.theta_im = torch.nn.Parameter(torch.nn.init.xavier_uniform_(theta_im, gain=self.magnitude))

        # n x 1
        lambda_re = torch.empty( (self.shape2), requires_grad=True, device=device )
        self.lambda_re = torch.nn.Parameter(torch.nn.init.uniform_( lambda_re , a=0.0, b=0.1))
        lambda_im = torch.empty( (self.shape2), requires_grad=True, device=device)
        self.lambda_im = torch.nn.Parameter(torch.nn.init.uniform_( lambda_im , a=0.0, b=0.1))

        # 1 x 1
        #delta = mx.sym.Variable('delta%d' % layer_num, shape = shape2, init = mx.init.Uniform(scale= 1.5))
        delta = torch.empty( (self.shape2), requires_grad=True, device=device) 
        self.delta = torch.nn.Parameter(torch.nn.init.uniform_( delta , b=1.5))
        if no_bias:
            self.lambda_c = 0.0
        else:
            self.lambda_c = 1.0
        self.complex_input = complex_input
    def forward(self, data):
        fi_in = data #QubitPhaseLimitLayer(data)
        if self.complex_input:
            x_re, x_im = fi_in[:][0], fi_in[:][1]
        else:
            x_re, x_im = get_re_im_from_fi(fi_in,1.0)
#         print(x_re.shape,x_im.shape, self.theta_re.shape)
        # k x n             # k x m  # m x n
#         print(x_re.shape, self.theta_re.shape, x_im.shape, self.theta_im.shape)
        x_re_theta_re = torch.tensordot(x_re, self.theta_re, dims=2)
        x_re_theta_im = torch.tensordot(x_re, self.theta_im, dims=2)
        x_im_theta_re = torch.tensordot(x_im, self.theta_re, dims=2)
        x_im_theta_im = torch.tensordot(x_im, self.theta_im, dims=2)

        # k x n
        x_theta_re = x_re_theta_re - x_im_theta_im
        x_theta_im = x_re_theta_im + x_im_theta_re

        # k x n            # n x 1
#         u_re = torch.broadcast_tensors(torch.add(x_theta_re, - lambda_c * lambda_re))
#         u_im = torch.broadcast_tensors(torch.add(x_theta_im, - lambda_c * lambda_im))
        #print(x_theta_re.shape, lambda_c, lambda_re.shape)
        u_re = torch.add(x_theta_re, - self.lambda_c * self.lambda_re)
        u_im = torch.add(x_theta_im, - self.lambda_c * self.lambda_im)
#         print(type(u_re))
        # k x n
        #print('Variant:',variant)
        if self.variant == 1: # Tricky arg
            arg = -1 * torch.atan( u_re ) #/ ((u_re) + sign_u_re * eps))

            # n x 1
            sigma = (np.pi / 2) * sigmoid(self.delta)

            # k x n
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 2: # not good
            sig_u_re = sigmoid(u_re)
            sig_u_im = sigmoid(u_im)
            y = sig_u_im
        elif self.variant == 3: # Honest arg
            sign_u_re = torch.sign(u_re,)
            u_im_not_zero = torch.abs(torch.sign(u_im))
            u_re_sign = torch.sign(u_re)
            numerator = torch.sqrt(torch.pow(u_im, 2) + torch.pow(u_re,  2)) - u_re
            arg = -1 * (2 * u_im_not_zero * torch.atan(numerator / u_im) + (np.pi/2 - np.pi/2 * u_re_sign) * (1-u_im_not_zero) )

            # n x 1
            sigma = (np.pi / 2) * sigmoid(self.delta)
            # k x n
            y = torch.add(arg, sigma)
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            #print(type(y))
        elif self.variant == 4: # Tricky arg with Gaussian
            arg = -1 * torch.atan(u_im ) #/ ((u_re) + sign_u_re * eps))

            # n x 1
            # Gaussian here is from the paper: "Qubit Neural Tree Network With Applicationsin Nonlinear System Modeling", doi:10.1109/ACCESS.2018.2869894
            sigma = (np.pi / 2) * gaussian(self.delta)

            # k x n
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 5: # Honest arg with Gaussian
            sign_u_re = torch.sign(u_re)
            u_im_not_zero = torch.abs(torch.sign(u_im))
            u_re_sign = torch.sign(u_re)
            numerator = torch.sqrt(torch.pow(u_im, 2) + torch.pow(u_re, 2)) - u_re
            arg = -1 * (2 * u_im_not_zero * torch.atan(numerator / u_im) + (np.pi/2 - np.pi/2 * u_re_sign) * (1-u_im_not_zero) )

            # n x 1
            # Gaussian here is from the paper: "Qubit Neural Tree Network With Applicationsin Nonlinear System Modeling", doi:10.1109/ACCESS.2018.2869894
            sigma = (np.pi / 2) * gaussian(self.delta)

            # k x n
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 6: # Tricky arg with Gaussian on trainable parameters
            arg = -1 * torch.atan(u_re ) #/ ((u_re) + sign_u_re * eps))

            # n x 1
            # Gaussian here is from the paper: "Qubit Neural Tree Network With Applicationsin Nonlinear System Modeling", doi:10.1109/ACCESS.2018.2869894
            #a = mx.sym.Variable('a%d' % layer_num, shape = shape2, init = mx.init.Uniform(scale= 1.0))
            a = torch.empty( (self.shape2), requires_grad=True, device=device)
            torch.nn.init.uniform_( a , b=1.0 )
            #b = mx.sym.Variable('b%d' % layer_num, shape = shape2, init = mx.init.Uniform(scale= 1.0))
            b = torch.empty( (self.shape2), requires_grad=True, device=device)
            torch.nn.init.uniform_( b , b=1.0 )
            sigma = (np.pi / 2) * gaussian(self.delta, a=a, b=b)

            # k x n
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 7: # Honest arg with Gaussian on trainable parameters
            sign_u_re = torch.sign(u_re)
            u_im_not_zero = torch.abs(torch.sign(u_im))
            u_re_sign = torch.sign(u_re)
            numerator = torch.sqrt(torch.pow(u_im, 2) + torch.pow(u_re, 2)) - u_re
            arg = -1 * (2 * u_im_not_zero * torch.atan(numerator / u_im) + (np.pi/2 - np.pi/2 * u_re_sign) * (1-u_im_not_zero) )

            # n x 1
            # Gaussian here is from the paper: "Qubit Neural Tree Network With Applicationsin Nonlinear System Modeling", doi:10.1109/ACCESS.2018.2869894
            #a = mx.sym.Variable('a%d' % layer_num, shape = shape2, init = mx.init.Uniform(scale= 1.0))
            a = torch.empty( (self.shape2), requires_grad=True, device=device)
            torch.nn.init.uniform_( a , b=1.0 )
            #b = mx.sym.Variable('b%d' % layer_num, shape = shape2, init = mx.init.Uniform(scale= 1.0))
            b = torch.empty( (self.shape2), requires_grad=True, device=device)
            torch.nn.init.uniform_( b , b=1.0 )
            sigma = (np.pi / 2) * gaussian(self.delta, a=a, b=b)

            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 8:
            arg = torch.atan(u_re ) #/ ((u_re) + sign_u_re * eps))
            # modification due to: "Chaotic Time Series Prediction by Qubit Neural Networkwith Complex-Valued Representation",
            # Taisei Ueguchi, Nobuyuki Matsui, and Teijiro Isokawa
            # Proceedings of the SICE Annual Conference 2016Tsukuba, Japan, September 20-23, 2016
            # URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7749232
            #arg = torch.broadcast_tensors(torch.mul(arg,1-2*sigmoid(self.delta)))
            arg = torch.mul(arg,1-2*sigmoid(self.delta))

            # n x 1
            sigma = (np.pi / 2) * sigmoid(self.delta)

            # k x n
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)
        elif self.variant == 9:
            sign_u_re = torch.sign(u_re)
            u_im_not_zero = torch.abs(torch.sign(u_im))
            u_re_sign = torch.sign(u_re)
            numerator = torch.sqrt(torch.pow(u_im, 2) + torch.pow(u_re, 2)) - u_re
            arg = (2 * u_im_not_zero * torch.atan(numerator / u_im) + (np.pi/2 - np.pi/2 * u_re_sign) * (1-u_im_not_zero) )
            # modification due to: "Chaotic Time Series Prediction by Qubit Neural Networkwith Complex-Valued Representation",
            # Taisei Ueguchi, Nobuyuki Matsui, and Teijiro Isokawa
            # Proceedings of the SICE Annual Conference 2016Tsukuba, Japan, September 20-23, 2016
            # URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7749232
            #arg = torch.broadcast_tensors(torch.mul(arg,1-2*sigmoid(self.delta)))
            arg = torch.mul(arg,1-2*sigmoid(self.delta))
            # n x 1
            sigma = (np.pi / 2) * sigmoid(self.delta)

            # k x n        
            #y = torch.broadcast_tensors(torch.add(arg, sigma))
            y = torch.add(arg, sigma)

        out_coeff = 2.0

        # k x n
        if self.out_layer:
            fi_out = torch.pow(torch.sin(y), self.power) * out_coeff
        else:
            fi_out = y # mx.sym.cos(y).__pow__(power) * 1
        return fi_out
        


# In[5]:


class ComplexConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, complex_input=False):
        super(ComplexConv,self).__init__()
        self.padding = padding
        self.complex_input = complex_input
        ## Model components
        self.conv_re = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
#         real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
#         imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        if self.complex_input:
            x_re, x_im = x[:][0], x[:][1]
        else:
            x_re, x_im = get_re_im_from_fi(x,1.0)
        real = self.conv_re(x_re) - self.conv_im(x_im)
        imaginary = self.conv_re(x_re) + self.conv_im(x_im)
        output = torch.stack((real,imaginary),dim=0)
        return output


# In[6]:


def ComplexRelu(x):
    x_re, x_im = x[:][0], x[:][1]
    real = torch.nn.functional.relu(x_re)
    imaginary = torch.nn.functional.relu(x_im)
    output = torch.stack((real,imaginary),dim=0)
    return output


# In[7]:


class ComplexPooling(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(ComplexPooling,self).__init__()
        ## Model components
        self.pool = torch.nn.AvgPool2d(kernel_size, stride)
        
    def forward(self, x):
        x_re, x_im = x[:][0], x[:][1]
        real = self.pool(x_re)
        imaginary = self.pool(x_im)
        output = torch.stack((real,imaginary),dim=0)
        return output

def train_model( model, trainset, criterion, optimizer, testset=None, epoch=20 ):
    prediction = None
    target = None
    train_losses = []
    test_losses = []
    test_loss = 0
    train_acc = []
    test_acc = []
    correct = 0
    total = 0
    acc = 0
    for t in range(epoch):
        losses = []
        best = 100
        log_interval = 10
        # Forward pass: Compute predicted y by passing x to the model
        for batch_idx, (data, target) in enumerate(trainset):
            y_pred = model(data.to(dev))
            _, predicted = torch.max(y_pred.data, 1)
            loss = criterion(y_pred, target.to(dev).type(torch.long))
            losses.append(loss.item())
            if best > loss.item():
                best = loss.item()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            #print(loss)
            loss.backward()
            optimizer.step()
#             if batch_idx % log_interval == 0:
#                 train_losses.append(loss.item())
#                 train_counter.append(
#                     (batch_idx*64) + ((t-1)*len(trainset.dataset)))
            total += target.size(0)
            correct += (predicted == target.to(dev)).sum().item()
        train_losses.append(np.average(losses))
        acc = (100 * correct / total)
        train_acc.append(acc)
        print('Train set: Epoch: {}, Best loss of batch: {:.4f}, Avg. loss: {:.4f} and Accuracy of the model: {:.4f}'.format( (t+1), best, np.average(losses), acc))
        if testset is not None:
            prediction, target, test_loss, acc = test_model( model, testset, criterion)
            test_losses.append(test_loss)
            test_acc.append(acc)
#         print(t, np.average(losses))
    return prediction, target, train_losses, test_losses, train_acc, test_acc


# In[10]:


def test_model( model, testset, criterion ):
    correct = 0
    total = 0
    best = 100
    losses = []
    prediction = None
    real = None
    acc  = 0
    try:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(testset):
                outputs = model(data.to(dev))
                _, predicted = torch.max(outputs.data, 1)
                if prediction is not None:
                    prediction = torch.cat((prediction, predicted))
#                     print('Appending prediction.')
                else:
                    prediction = predicted
                if real is not None:
                    real = torch.cat((real, target.type(torch.long)))
                else:
                    real = target.type(torch.long)
                loss = criterion(outputs, target.to(dev).type(torch.long))
                losses.append(loss.item())
                if best > loss.item():
                    best = loss.item()    
                total += target.size(0)
                correct += (predicted == target.to(dev)).sum().item()
        acc = (100 * correct / total)
        print('Test set: Best loss of batch: {:.4f} , Avg. loss: {:.4f} and Accuracy of the model: {:.4f}'.format( best, np.average(losses),acc) )
#         print('Accuracy of the model: %d %%' % (
#             100 * correct / total))
        return prediction, real, np.average(losses), acc
    except Exception as e:
        print('Error in calculating accuracy.')
        e.printStackTrace()


# In[11]:


def plot_loss( x1, y1, x2, y2 , figure_name=None):
    '''
    x1 -> train_counter
    x2 -> test_counter
    y1 -> train_losses
    y2 -> test_losses
    '''
    fig = plt.figure()
    plt.plot(x1, y1, color='blue')
#     plt.scatter(test_counter, test_losses, color='red')
    plt.plot(x2, y2, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')   
    if figure_name is not None:
        plt.savefig(figure_name)
    fig


# In[12]:


def plot_acc( x1, y1, x2, y2 , figure_name=None):
    '''
    x1 -> train_counter
    x2 -> test_counter
    y1 -> train_accuracy
    y2 -> test_accuracy
    '''
    fig = plt.figure()
    plt.plot(x1, y1, color='blue')
#     plt.scatter(test_counter, test_losses, color='red')
    plt.plot(x2, y2, color='red')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')   
    if figure_name is not None:
        plt.savefig(figure_name)
    fig


# In[13]:


def plot_prediction(target,prediction):
    labels = [i+1 for i in range(len(target))]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, target, width, label='Target Label')
    rects2 = ax.bar(x + width/2, prediction, width, label='Prediction Label')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Label')
    ax.set_title('Target vs Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param






