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

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=True, num_workers=1,
                                          pin_memory=True)



def init_weights(shape):
    w = torch.randn(size=shape)*0.01
    w.requires_grad = True
    return w

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
    def __init__(self, params, lr=1e-3, alpha=0.9, eps=1e-8):
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
        phi = np.random.binomial(1, p_drop, (X.shape[0], X.shape[1]))
        phi[phi == 1] = 1.0 / p_drop
        phi = torch.from_numpy(phi).float()
        X = X * phi
    return X


def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden, train):
    X = dropout(X, p_drop_input)
    h = rectify(X @ w_h)
    if train:
        h = dropout(h, p_drop_hidden)
    h2 = rectify(h @ w_h2)
    if train:
        h2 = dropout(h2, p_drop_hidden)
    pre_softmax = h2 @ w_o
    return pre_softmax #.transpose(0,1)


w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

optimizer = RMSprop([w_h, w_h2, w_o])




# put this into a training loop over 100 epochs
for (_, (X, y)) in enumerate(dataloader, 0):
    noise_py_x = model(X.reshape(mb_size, 784), w_h, w_h2, w_o, 0.8, 0.7, False)
    cost = torch.nn.functional.cross_entropy(noise_py_x, y)
    cost.backward()
    print("Loss: {}".format(cost))
    optimizer.step()

for (_, (X, y)) in enumerate(test_loader, 0):
    noise_py_x = model(X.reshape(test_batch_size, 784), w_h, w_h2, w_o, 0.8, 0.7, False)
    cost = torch.nn.functional.cross_entropy(noise_py_x, y)
    print("Testloss: {}".format(cost))
