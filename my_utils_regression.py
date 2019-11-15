from typing import List, Iterable
import torch
import numpy as np
from my_utils import print_and_log
from BaseCNN import BaseConvNet

def get_next_chkpt(checkpoints_list: List[int]) -> int:
    for chkpt in checkpoints_list:
        yield chkpt

def train(model, 
          optimizer: torch.optim.Optimizer, 
          loss: torch.nn.modules.loss._Loss, 
          data: torch.utils.data.DataLoader, 
          epochs: int = 10,
          cuda: bool = True,
          logfile = None,
          checkpoints: List[int] = None,
          checkpoints_folder: str = "") -> List[float]:
    model = model.cuda() if cuda == True else model
    
    if checkpoints is not None:
        chkpts = get_next_chkpt(checkpoints)
    
    chkpt = next(chkpts)
    chkpt_counter = 0

    lloss = []
    for epoch in range(1, epochs+1):
        if epoch != 1:
            print_and_log(("",), logfile)

        print_and_log((f"Epoch #{epoch}:", "-" * 15), logfile)

        model.train()
        loss_sum = 0
        for i, (x,y) in enumerate(data):
            if cuda == True:
                x, y = x.cuda(), y.cuda()
              
            pred = model(x)
            l = loss(pred.view(-1), y)
            l.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += l.item()
            print_and_log((f"Batch #{i}\tLoss: {l}",), logfile)
        
        avg_loss = loss_sum / (i+1)
        lloss.append(avg_loss)
        
        print_and_log((f"Avg Training Loss: {avg_loss}",), logfile)

        if epoch == chkpt:
            save_model(model, epoch, checkpoints_folder)
            chkpt_counter += 1
            if chkpt_counter < len(checkpoints):
                chkpt = next(chkpts)
    
    return lloss


def save_model(model: BaseConvNet, epoch:int, folder: str) -> None:
    if folder[-1] != '/':
        folder = folder + '/'

    filename = f"{folder}{model.name}_{epoch}.pth"
    torch.save({"epochs": epoch,
                "state_dict": model.state_dict()}, 
                filename)

def eval(model, 
         data: torch.utils.data.DataLoader,
         loss: torch.nn.modules.loss._Loss, 
         cuda: bool = True):
    if cuda == True:
        model = model.cuda()

    model.eval()
    with torch.set_grad_enabled(False):

        predictions = []
        sum_loss = 0
        i = 0
        for x, y in data:
            if cuda == True:
                x, y = x.cuda(), y.cuda()

            pred = model(x)
            l = loss(pred.view(-1), y)
            
            sum_loss += l.item()
            predictions.append(pred)
            i += 1
    
    return predictions, sum_loss/i
