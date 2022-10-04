import torch.nn as nn
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self, cnn, wGANLoss=1., wFeatureMatch=10., wPerceptual=10., wKLDivergence=0.05):
        super().__init__()
        self.wGANLoss      = wGANLoss
        self.wFeatureMatch = wFeatureMatch
        self.wPerceptual   = wPerceptual
        self.wKLDivergence = wKLDivergence

        self.cnn        = cnn
        self.cnnWeights = [1. / 32, 1. / 16, 1. / 8, 1. / 4, 1.]

    def forward(self, realImage, realOutsList, fakeImage, fakeOutsList, mu, logVar):
        # lossGAN = self.GetGANLoss([outs[-1] for outs in fakeOutsList])
        # lossFM  = self.GetFeatureMatchLoss(realOutsList, fakeOutsList)
        # lossPer = self.GetVgg19PerceptualLoss(realImage, fakeImage)
        # lossDiv = self.GetKLDivergenceLoss(mu, logVar)
        return (
            self.wGANLoss      * self.GetGANLoss([outs[-1] for outs in fakeOutsList]) + 
            self.wFeatureMatch * self.GetFeatureMatchLoss(realOutsList, fakeOutsList) +
            self.wPerceptual   * self.GetVgg19PerceptualLoss(realImage, fakeImage)    + 
            self.wKLDivergence * self.GetKLDivergenceLoss(mu, logVar)                
        )

    def GetGANLoss(self, fakeFinalOuts):
        return -sum(o.mean() for o in fakeFinalOuts)
    
    def GetFeatureMatchLoss(self, realOutsList, fakeOutsList):
        return sum(sum(F.l1_loss(oF, oR) for oR, oF in zip(realOuts[:-1], fakeOuts[:-1])) for realOuts, fakeOuts in zip(realOutsList, fakeOutsList)) / len(realOutsList)
    
    def GetVgg19PerceptualLoss(self, realImage, fakeImage):
        return sum(w * F.l1_loss(pF, pR) for w, pR, pF in zip(self.cnnWeights, self.cnn(realImage), self.cnn(fakeImage)))
    
    def GetKLDivergenceLoss(self, mu, logVar):
        return 0.5 * (logVar.exp() - 1 - logVar + mu.pow(2)).sum()
    

class DiscriminatorLoss(nn.Module):
    def forward(self, realOutsList, fakeOutsList):
        return (
            self.GetRealHingeLoss([outs[-1] for outs in realOutsList]) + 
            self.GetFakeHingeLoss([outs[-1] for outs in fakeOutsList])
        ) * 0.5 / len(realOutsList)
    
    def GetRealHingeLoss(self, finalOuts):
        return sum(F.relu(1 - o).mean() for o in finalOuts)
    
    def GetFakeHingeLoss(self, finalOuts):
        return sum(F.relu(1 + o).mean() for o in finalOuts)