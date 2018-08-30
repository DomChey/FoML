"""
Training, evaluation and testing routines for the BuddiesNet

@author: Dominique Cheray
"""


# necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from piecesLoader import get_train_and_valid_loader, get_test_loader
from buddiesNet import BuddiesNet
from tqdm import tqdm

# define some usefull globals
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
BATCH_SIZE = 960
# best validation accuracy
BEST_ACC = 0
# start from 0 or last checkpoint
START_EPO = 0
# parameters for the optimizer
MOMENTUM = 0.9
LR = 0.01


def train(epoch, model, train_loader, optimizer, criterion, log):
    """Training method for the BuddiesNet

    Args:
        epoch:           Idx of training epoch
        model:           The network to train
        train_loader:    The dataloader for the training pieces
        optimizer:       The optimizer for the network
        criterion:       The criterion for the network
        log:             The logfile to log accuracy and loss"""

    log.write("Epoch: {}\n".format(epoch))
    if epoch % 10 == 0:
        print("Epoch: {}".format(epoch))

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for (data, target) in tqdm(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction.float(), target)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        _, pred = prediction.max(1)
        total = total + target.size(0)
        correct = correct + pred.eq(target).sum().item()

    log.write("Training: Loss: {:.2f} | Acc: {:.2f}%\n".format(
        (train_loss/len(train_loader)), (correct/total*100)))
    if epoch % 10 == 0:
        print("Training: Loss: {:.2f} | Acc: {:.2f}%".format(
        (train_loss/len(train_loader)), (correct/total*100)))



def validation(epoch, model, valid_loader, criterion, log, optimizer):
    """Validation method for the BuddiesNet

    Args:
        epoch:           Idx of the validation epoch
        model:           The network to validate
        valid_loader:    The validation loader
        criterion:       The criterion for the network
        log:             The logfile to log accuracy and loss
        optimizer:       The optimizer for the model, will be needed if model is saved
    """

    global BEST_ACC
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prediction = model(data)
            loss = criterion(prediction.float(), target)

            valid_loss = valid_loss + loss.item()
            _, pred = prediction.max(1)
            total = total + target.size(0)
            correct = correct + pred.eq(target).sum().item()

        log.write("Validation: Loss: {:.2f} | Acc: {:.2f}%\n".format(
            (valid_loss/len(valid_loader)), (correct/total*100)))
        if epoch % 10 == 0:
            print("Validation: Loss: {:.2f} | Acc: {:.2f}%".format(
            (valid_loss/len(valid_loader)), (correct/total*100)))

    # if the model performs well, save it
    accuracy = correct/total*100
    if accuracy > BEST_ACC:
        log.write("Saving model")
        print("Saving model")
        # push model to cpu before saving it
        tmp_modle = model.to('cpu')
        state = {
            'model': tmp_model.state.dict(),
            'accuracy': accuracy,
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, 'ckpt.t7')
        BEST_ACC = accuracy


def resume_from_checkpoint(model, checkpoint, optimizer):
    """Method to load a saved checkpoint and initialize model and optimizer
    whith the saved values

    Args:
        model:         The model to add the saved parameters to
        checkpoint:    Which checkpoint file to load
        optimizer:     The optimizer to add the saved parameters to"""

    global BEST_ACC
    global START_EPO
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    BEST_ACC = ckpt['accuracy']
    START_EPO = ckpt['epoch']

    return model, optimizer


def train_dat_net(start_epoch, model):
    """Training and validation routine for the BuddiesNet

    Args:
        epoch:    Idx of start epoch
        model:    Network to train and validate"""

    # instantiate all that is needed for the training and validation
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    train_loader, validation_loader = get_train_and_valid_loader(BATCH_SIZE, USE_CUDA,
                                                                 random_seed=1)
    log = open("logfile.txt", "w")
    # now start training and validation
    for epoch in tqdm(range(start_epoch, start_epoch + 100)):
        train(epoch, model, train_loader, optimizer, criterion, log)
        validation(epoch, model, validation_loader, criterion, log, optimizer)
    log.close()


def test_dat_net(model):
    """Testing method for the BuddiesNet

    Args:
        model:    The Network to test"""

    print("Testing")
    model = model.to(DEVICE)
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    test_log = open("testLog.txt", "w")

    test_loader = get_test_loader(960, USE_CUDA)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (data, target) in tqdm(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prediction = model(data)
            loss = criterion(prediction.float(), target)

            test_loss = test_loss + loss.item()
            _, pred = prediction.max(1)
            total = total + target.size(0)
            correct = correct + pred.eq(target).sum().item()

        print("Testing: Loss: {:.2f} | Acc: {:.2f}".format((test_loss/len(test_loader)),
                                                           (correct/total*100)))
        test_log.write("Testing: Loss: {:.2f} | Acc: {:.2f}".format(
            (test_loss/len(test_loader)), (correct/total*100)))


model = BuddiesNet()
train_dat_net(START_EPO, model)
test_dat_net(model)
