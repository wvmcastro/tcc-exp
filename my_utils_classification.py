from typing import Tuple, List
import torch
import numpy as np

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.argmax(pred, dim=1) == y).float().mean().item()

def per_class_accuracy(pred, y, nclasses) -> List:
    accs = [0.0] * nclasses
    ns = [0] * nclasses
    for pred, y in zip(pred, y):
        predictions = torch.argmax(pred, dim=1)

        for i in range(len(y)):
            accs[y[i]] += float(predictions[i] == y[i])
            ns[y[i]] += 1

    for i, n in enumerate(ns):
        if n != 0:
            accs[i] /= n
        else:
            print(i, "n == 0")    
    return accs

def train(model, 
          optimizer: torch.optim.Optimizer, 
          loss: torch.nn.modules.loss._Loss, 
          data: torch.utils.data.DataLoader, 
          epochs: int = 10,
          cuda: bool = True) -> Tuple[List[float], List[float]]:
    model = model.cuda() if cuda == True else model
    
    llloss = []
    llacc = []
    for epoch in range(epochs):
        lloss = []
        lacc = []
        model.train()
        for x,y in data:
            if cuda == True:
                x, y = x.cuda(), y.cuda()
              
            pred = model(x)
            l = loss(pred.view(-1), y)

            lloss.append(l.item())
            lacc.append(accuracy(pred, y))

            l.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        llloss.append(np.mean(lloss))
        llacc.append(np.mean(lacc))
        print("Epoch: %r\t AvgLoss: %r\t AvgAcc: %r\t" %\
              (epoch, llloss[-1], llacc[-1]))
    
    return (llloss, llacc)

def test(model, 
         data: torch.utils.data.DataLoader, 
         cuda: bool = True) -> List[float]:

    if cuda == True:
        model = model.cuda()

    model.eval()
    with torch.set_grad_enabled(False):
        lacc = []
        preds = []
        ys = []
        for x, y in data:
            if cuda == True:
                x = x.cuda()
                y = y.cuda()

            pred = model(x)
            
            lacc.append(accuracy(pred, y))
            preds.append(pred)
            ys.append(y)

    print("Per Class acc: ", per_class_accuracy(preds, ys, 4))

    return lacc, pred




