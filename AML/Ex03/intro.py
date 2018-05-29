import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.optim
import torch.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d



mb_size = 100 # mini-batch size of 100
test_batch_size = 1000


trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])


dataset = dset.MNIST("./", download = True,
                     train = True,
                     transform = trans)

test_dataset = dset.MNIST("./", download=True,
                          train=False,
                          transform = trans)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=mb_size,
                                         shuffle=True, num_workers=1,
                                         pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mb_size,
                                          shuffle=True, num_workers=1,
                                          pin_memory=True)


def init_weights(shape):
    # xavier initialization (a good initialization is important!)
    # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    fan_in = shape[0]
    fan_out = shape[1]
    variance = 2.0/(fan_in + fan_out)
    w = torch.randn(size=shape)*np.sqrt(variance)
    w.requires_grad = True
    return w


def init_alpha(shape):
    a = torch.zeros(size=shape)
    a.requires_grad = True
    return a

def rectify(X):
    return torch.max(torch.zeros_like(X), X)


# you can also use torch.nn.functional.softmax on future sheets
def softmax(X):
    c = torch.max(X, dim=1)[0].reshape(mb_size, 1)
    # this avoids a blow up of the exponentials
    # but calculates the same formula
    stabelized = X-c
    exp = torch.exp(stabelized)
    return exp/torch.sum(exp, dim=1).reshape(mb_size, 1)


# this is an example as a reduced version of the pytorch internal RMSprop optimizer
class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(-group['lr'], grad, avg)


def dropout(X, p_drop=1.):
    if p_drop > 0 and p_drop < 1:
        phi = np.random.binomial(1, p_drop, X.shape)
        phi[phi == 1] = 1.0 / p_drop
        phi = torch.from_numpy(phi).float()
        X = X * phi
    return X


def PRelu(X, a):
    X_copy = X.clone()
    tmp = torch.mul(X, a)
    X_copy[X <= 0] = tmp[X <= 0]
    return X_copy


def model(X, w_h, w_h2, w_o, a, a2, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(X @ w_h)
    #h = PRelu(X @w_h, a)
    h = dropout(h, p_drop_hidden)
    h2 = rectify(h @ w_h2)
    #h2 = PRelu(h @ w_h2, a2)
    h2 = dropout(h2, p_drop_hidden)
    pre_softmax = h2 @ w_o
    return pre_softmax #.transpose(0,1)


w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))
a = init_weights((mb_size, 625))
a2 = init_weights((mb_size, 625))

optimizer = RMSprop([w_h, w_h2, w_o])




# put this into a training loop over 100 epochs
for i in range(1):
    print("Epoch: {}".format(i+1))
    for (_, (X, y)) in enumerate(dataloader, 0):
        noise_py_x = model(X.reshape(mb_size, 784), w_h, w_h2, w_o, a, a2, 0.8, 0.7)
        cost = torch.nn.functional.cross_entropy(noise_py_x, y)
        cost.backward()
        if i % 10 == 0:
            print("Loss: {}".format(cost))
        optimizer.step()

    for (_, (X, y)) in enumerate(test_loader, 0):
        noise_py_x = model(X.reshape(mb_size, 784), w_h, w_h2, w_o, a, a2, 0.0, 0.0)
        cost = torch.nn.functional.cross_entropy(noise_py_x, y)
        if i % 10 == 0:
            print("Testloss: {}".format(cost))




def conv_model(X, conv_w1, conv_w2, conv_w3, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    conv1 = rectify(conv2d(X, conv_w1))
    print("conv1: {}".format(conv1.shape))
    pool1 = max_pool2d(conv1, (2, 2))
    print("pool1: {}".format(pool1.shape))
    drop1 = dropout(pool1, p_drop_hidden)
    conv2 = rectify(conv2d(drop1, conv_w2))
    print("conv2: {}".format(conv2.shape))
    pool2 = max_pool2d(conv2, (2, 2))
    print("pool2: {}".format(pool2.shape))
    drop2 = dropout(pool2, p_drop_hidden)
    conv3 = rectify(conv2d(drop2, conv_w3))
    print("conv3: {}".format(conv3.shape))
    pool3 = max_pool2d(conv3, (2, 2))
    print("pool3: {}".format(pool3.shape))
    drop3 = dropout(pool3, p_drop_hidden)
    h2 = rectify(torch.reshape(drop3, (mb_size, 128)) @ w_h2)
    pre_softmax = h2 @ w_o
    return pre_softmax

   # torch.reshape()

conv_w1 = init_weights((32, 1, 5, 5))
conv_w2 = init_weights((64, 32, 5, 5))
conv_w3 = init_weights((128, 64, 2, 2))
w_h2 = init_weights((128 ,625))
w_o = init_weights((625, 10))

conv_optimizer = RMSprop([conv_w1, conv_w2, conv_w3, w_h2, w_o])

for i in range(1):
    print("Epoch: {}".format(i+1))
    for (_, (X, y)) in enumerate(dataloader, 0):
        noise_py_x = conv_model(X.view(-1, 1, 28, 28), conv_w1, conv_w2, conv_w3, w_h2, w_o, 0.8, 0.7)
        cost = torch.nn.functional.cross_entropy(noise_py_x, y)
        cost.backward()
        if i % 10 == 0:
            print("Loss: {}".format(cost))
        conv_optimizer.step()

    for (_, (X, y)) in enumerate(test_loader, 0):
        noise_py_x = conv_model(X.view(-1, 1, 28, 28), conv_w1, conv_w2, conv_w3, w_h2, w_o, 0.0, 0.0)
        cost = torch.nn.functional.cross_entropy(noise_py_x, y)
        if i % 10 == 0:
            print("Testloss: {}".format(cost))
