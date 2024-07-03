# Wonsang Yun, Dept. of mathmetics, Yonsei Univ.
# DEEP LEARNING (STA3140.01-00) Final project 2024/05/07~

import torch.optim as optim
import torch.nn as nn
import torch

from resnet import ResNet18
from dataLoader import trainLoader, testLoader
from SAM import SAM
from Adversarial import fgsm_attack
from Blur import Blur

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(args, dataloader, model, loss_fn, optimizer, epsilon):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        if args.method == 'Adam':
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        elif args.method == 'Blur':
            blur_X = Blur(X, epsilon, device)
            pred = model(blur_X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        elif args.method == 'Adversarial':
            X.requires_grad = True
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()
            X_grad = X.grad.data
            attacked_X = fgsm_attack(X, epsilon, X_grad)

            # Train attacked_X
            pred = model(attacked_X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        elif args.method == 'SAM':
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()

            def closure():
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()

                return loss
            #END of closure()

            loss.backward()
            optimizer.step(closure)


        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, l):
    size = l
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, 100*correct

def train_test(args):

    print(device)
    # Hyperparameter
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    rho = args.rho

    trainloader, valloader, split = trainLoader(batch_size)
    testTloader, lT = testLoader('T', batch_size)
    testUloader, lU = testLoader('U', batch_size)
    testSloader, lS = testLoader('S', batch_size)
    testIloader, lI = testLoader('I', batch_size)
    testOloader, lO = testLoader('O', batch_size)
    testMloader, lM = testLoader('M', batch_size)
    resnet18 = ResNet18(3).to(device)

    # Loss function, Optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    if args.method == 'Adam' or args.method == 'Adversarial' or args.method == 'Blur':
        optimizer = optim.Adam(params=resnet18.parameters(), lr=lr)
    
    if args.method == 'SAM' or args.method == 'AdversarialSAM':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(resnet18.parameters(), base_optimizer, rho=rho, lr=lr)  #, momentum=0.9

    history_loss = []
    history_acc = []
    val_history_loss = []
    val_history_acc = []

    epsilon = np.linspace(start=args.init, stop=args.end, num=epochs)
    for i in range(epochs):
        print(f'Epoch {i+1}\n------------------------------')
        
        train_loop(args, trainloader, resnet18, loss_fn, optimizer, epsilon[i])
        print('Validation ', end = '')
        loss, acc = test_loop(valloader, resnet18, loss_fn, split)
        val_history_loss.append(loss)
        val_history_acc.append(acc)



        # TEST each testbed
        if (i + 1) % 5 == 0:
            print('Testing...')
            print()

            temp_loss = []
            temp_acc = []
            # Test T
            print('Test T', end = ' ')
            loss, acc = test_loop(testTloader, resnet18, loss_fn, lT)
            temp_loss.append(loss)
            temp_acc.append(acc)

            # Test U
            print('Test U', end = ' ')
            loss, acc = test_loop(testUloader, resnet18, loss_fn, lU)
            temp_loss.append(loss)
            temp_acc.append(acc)

            # Test S
            print('Test S', end = ' ')
            loss, acc = test_loop(testSloader, resnet18, loss_fn, lS)
            temp_loss.append(loss)
            temp_acc.append(acc)

            # Test I
            print('Test I', end = ' ')
            loss, acc = test_loop(testIloader, resnet18, loss_fn, lI)
            temp_loss.append(loss)
            temp_acc.append(acc)

            # Test O
            print('Test O', end = ' ')
            loss, acc = test_loop(testOloader, resnet18, loss_fn, lO)
            temp_loss.append(loss)
            temp_acc.append(acc)

            # Test M
            print('Test M', end = ' ')
            loss, acc = test_loop(testMloader, resnet18, loss_fn, lM)
            temp_loss.append(loss)
            temp_acc.append(acc)

            history_loss.append(temp_loss)
            history_acc.append(temp_acc)
    #for(i)

    return val_history_loss, val_history_acc, history_loss, history_acc