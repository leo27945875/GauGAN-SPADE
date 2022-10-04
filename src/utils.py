import os
import io
import random
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings

filterwarnings("ignore")

import torch
from gluoncv.utils.viz import get_color_pallete


def GPUToNumpy(tensor, reduceDim=None, isSqueeze=True):
    if type(tensor) is np.array or type(tensor) is np.ndarray:
        return tensor
    
    if isSqueeze:
        if reduceDim is not None:
            return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
        else:
            return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)
    
    else:
        if len(tensor.shape) == 3:
            return tensor.cpu().detach().numpy().transpose(1, 2, 0)
        elif len(tensor.shape) == 4:
            return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)


def MaskToOnehot(mask, c=150):
    h, w, device = mask.shape[0], mask.shape[1], mask.device
    onehot = torch.zeros([c, h, w], device=device)
    onehot.scatter_(0, mask.unsqueeze(0), torch.ones([1, h, w], device=device))
    return onehot


def SeedEverything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Denormalize(x, mean=0.5, std=0.5):
    return x * std + mean


def DetachAll(tensorList):
    tensorListNew = []
    while tensorList:
        tensor = tensorList.pop(0)
        if isinstance(tensor, torch.Tensor):
            tensorListNew.append(tensor.detach())
        elif hasattr(tensor, '__iter__'):
            tensorListNew.append(DetachAll(tensor))
        else:
            tensorListNew.append(tensor)
    
    return tensorListNew


def PrintTrainMessage(epoch, batch, maxEpoch, maxBatch, lossDis, lossGen):
    print(f'\r| Epoch {epoch :4d}/{maxEpoch :4d} | Batch {batch :4d}/{maxBatch :4d} | LossD {lossDis :.10f} | LossG {lossGen :.10f} |', end='')


def SaveCheckPoint(folder, filename, epoch, model, optimizer=None, scheduler=None, scaler=None):
    state = {
        'epoch'    : epoch,
        'model'    : model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else {},
        'scheduler': scheduler.state_dict() if scheduler else {},
        'scaler'   : scaler.state_dict() if scaler else {}
    }
    torch.save(state, os.path.join(folder, filename))
    return state


def LoadGenerator(folder, filename, gen):
    state = torch.load(os.path.join(folder, filename))
    gen.load_state_dict(state['model'])
    return gen


def GetColorMask(mask):
    mask = get_color_pallete(mask, dataset='ade20k')
    temp = io.BytesIO()
    mask.save(temp, 'png')
    return plt.imread(temp, 'jpg')


def CountLabel(mask, labelList):
    return {label: count for label in labelList if (count := (mask == label).sum().item())}


class LinearDecayScheduler:
    def __init__(self, optimizer, startEpoch, endEpoch):
        self.optimizer  = optimizer
        self.startEpoch = startEpoch
        self.endEpoch   = endEpoch
        self.initLR     = optimizer.param_groups[0]['lr']
    
    def Update(self, epoch):
        if epoch < self.startEpoch or epoch > self.endEpoch:
            return self.GetLR()
        
        newLR = self.initLR * (self.endEpoch - epoch)/(self.endEpoch - self.startEpoch)
        self.SetLR(newLR)
        return newLR

    def SetLR(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def GetLR(self):
        return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    import cv2

    inFile = r'D:\Downloads\AI\Image Manipulation Techniques and Visual Effects\Final\save\result10\synthesis\3\ADE_train_00011982_Test..png'
    outFile = r'D:\Downloads\AI\Image Manipulation Techniques and Visual Effects\Final\save\result10\synthesis\3\ADE_train_00011982_Color..png'

    mask = cv2.imread(inFile, cv2.IMREAD_GRAYSCALE)
    color = GetColorMask(mask)
    cv2.imwrite(outFile, color[:, :, :3] * 255)