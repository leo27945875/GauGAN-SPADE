import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def InitializeWeights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight.data, 0.02)
        if hasattr(module, 'bias'):
            nn.init.constant_(module.bias.data, 0.0)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Resize(nn.Module):
    def __init__(self, scale, mode="nearest"):
        super().__init__()
        self.scale = scale
        self.mode  = mode
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        pretrainedVGG19 = torchvision.models.vgg19(pretrained=True).features
        self.slice0 = torch.nn.Sequential(*[pretrainedVGG19[i] for i in range(2     )])
        self.slice1 = torch.nn.Sequential(*[pretrainedVGG19[i] for i in range(2 , 7 )])
        self.slice2 = torch.nn.Sequential(*[pretrainedVGG19[i] for i in range(7 , 12)])
        self.slice3 = torch.nn.Sequential(*[pretrainedVGG19[i] for i in range(12, 21)])
        self.slice4 = torch.nn.Sequential(*[pretrainedVGG19[i] for i in range(21, 30)])
        self.requires_grad_(False)

    def forward(self, x):
        h0 = self.slice0(x)
        h1 = self.slice1(h0)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h0, h1, h2, h3, h4


class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros([1, channels, 1, 1]))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class SpectralConv2d(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize=3, stride=1, padding=1):
        super().__init__()
        self.conv  = spectral_norm(nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding))
    
    def forward(self, x):
        return self.conv(x)


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel, isInstanceNorm=True, isDownSample=False):
        super().__init__()
        self.conv = nn.Sequential(
            SpectralConv2d(inChannel, outChannel),
            nn.AvgPool2d(2) if isDownSample else Identity(),
            nn.InstanceNorm2d(outChannel) if isInstanceNorm else Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.conv = nn.Sequential(
            SpectralConv2d(inChannel, outChannel, 3, 2, 1),
            nn.InstanceNorm2d(outChannel),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class SPADE(nn.Module):
    def __init__(self, maskScale, nClass, outChannel):
        super().__init__()
        self.convInit  = nn.Sequential(
            Resize(maskScale),
            SpectralConv2d(nClass, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convGamma = SpectralConv2d(128, outChannel)
        self.convBeta  = SpectralConv2d(128, outChannel)
        self.norm      = nn.InstanceNorm2d(outChannel, affine=False)
    
    def forward(self, x, onehot):
        conv0 = self.convInit(onehot)
        return self.convGamma(conv0) * self.norm(x) + self.convBeta(conv0)


class SPADE_ResBlock(nn.Module):
    def __init__(self, inChannel, outChannel, nClass, maskScale, isUpsample=True):
        super().__init__()
        self.isUpsample         = isUpsample
        self.isDifferentChannel = inChannel != outChannel

        self.norm0 = SPADE(maskScale, nClass, inChannel)
        self.conv0 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            SpectralConv2d(inChannel, outChannel)
        )

        self.norm1 = SPADE(maskScale, nClass, outChannel)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            SpectralConv2d(outChannel, outChannel)
        )

        self.normS = SPADE(maskScale, nClass, inChannel) if self.isDifferentChannel else None
        self.convS = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            SpectralConv2d(inChannel, outChannel)
        )  if self.isDifferentChannel else None

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2) if isUpsample else None
    
    def forward(self, x, onehot):
        h = self.conv0(self.norm0(x, onehot))
        h = self.conv1(self.norm1(h, onehot))

        if self.isDifferentChannel:
            h = h + self.convS(self.normS(x, onehot))
        else:
            h = h + x
        
        if self.isUpsample:
            return self.upsample(h)
        else:
            return h


class ImageEncoder(nn.Module):
    def __init__(self, noiseDim, imageSize, imageChannel=3, minChannel=64, maxChannel=512):
        super().__init__()
        nDownSampleLayer = int(np.log2(imageSize / 4))

        self.conv = nn.Sequential(*[
            EncoderConvBlock(self.GetInChannel (i, nDownSampleLayer, imageChannel, maxChannel),
                             self.GetOutChannel(i, nDownSampleLayer, minChannel  , maxChannel))
            for i in range(nDownSampleLayer)
        ])
        self.mu     = nn.Linear(4 * 4 * maxChannel, noiseDim)
        self.logVar = nn.Linear(4 * 4 * maxChannel, noiseDim)

        self.apply(InitializeWeights)
    
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        mu, logVar = self.mu(h), self.logVar(h)
        return self.SampleGaussian(mu, logVar), mu, logVar

    @staticmethod
    def GetInChannel(iDownSampleLayer, nDownSampleLayer, imageChannel, maxChannel):
        iBack = nDownSampleLayer - iDownSampleLayer - 1
        if   iBack <= 1           : return maxChannel
        elif iDownSampleLayer == 0: return imageChannel
        else                      : return maxChannel // 2 ** (iBack - 1)
    
    @staticmethod
    def GetOutChannel(iDownSampleLayer, nDownSampleLayer, minChannel, maxChannel):
        iBack = nDownSampleLayer - iDownSampleLayer - 1
        if   iBack <= 2: return maxChannel
        elif iBack >= 5: return minChannel
        else           : return maxChannel // 2 ** (iBack - 2)
    
    @staticmethod
    def SampleGaussian(mu, logVar):
        e = torch.randn_like(logVar, device=logVar.device)
        return mu + e * (logVar * 0.5).exp()


class Generator(nn.Module):
    def __init__(self, noiseDim, nClass, imageSize, imageChannel=3, maxChannel=1024, encoderMinChannel=64, encoderMaxChannel=512):
        super().__init__()
        nUpsample       = int(np.log2(imageSize / 4))
        finalOutChannel = self.GetOutChannel(nUpsample - 1, maxChannel)

        self.encoder = ImageEncoder(noiseDim, imageSize, imageChannel, encoderMinChannel, encoderMaxChannel)

        self.input  = nn.Sequential(nn.Linear(noiseDim, 16384))
        self.convs  = nn.ModuleList([
            SPADE_ResBlock(self.GetInChannel (i, maxChannel), 
                           self.GetOutChannel(i, maxChannel),
                           nClass, 
                           (4 * 2 ** i) / imageSize) for i in range(nUpsample)
        ])
        self.noises = nn.ModuleList([
            NoiseLayer(self.GetInChannel(i, maxChannel)) for i in range(nUpsample)
        ])
        self.output = nn.Sequential(
            NoiseLayer(finalOutChannel),
            nn.Conv2d(finalOutChannel, imageChannel, 3, 1, 1),
            nn.Tanh()
        )
        
        self.apply(InitializeWeights)
    
    def forward(self, image, onehot):
        noise, mu, logVar = self.encoder(image)
        h = self.input(noise).view(noise.size(0), -1, 4, 4)
        for convLayer, noiseLayer in zip(self.convs, self.noises):
            h = convLayer(noiseLayer(h), onehot)
        
        return self.output(h), mu, logVar

    @staticmethod
    def GetInChannel(iUpsample, maxChannel):
        return maxChannel if iUpsample <= 3 else maxChannel // 2 ** (iUpsample - 3)
    
    @staticmethod
    def GetOutChannel(iUpsample, maxChannel):
        return maxChannel if iUpsample <= 2 else maxChannel // 2 ** (iUpsample - 2)


class Discriminator(nn.Module):
    def __init__(self, inChannel, minChannel=64, maxChannel=512, nDownSampleLayer=3, nDiscriminator=3):
        super().__init__()
        self.discriminators  = nn.ModuleList([self.GetOneDiscriminator(inChannel, minChannel, maxChannel, nDownSampleLayer) for _ in range(nDiscriminator)])
        self.downSampleInput = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

        self.apply(InitializeWeights)
    
    def forward(self, x, onehot):
        img, outs, n = torch.concat([x, onehot], dim=1), [], len(self.discriminators)
        for i, dis in enumerate(self.discriminators, 1):
            outs.append(self.GetOneDiscriminatorOuts(dis, img))
            if i < n:
                img = self.downSampleInput(img)
        
        return outs
    
    @staticmethod
    def GetOneDiscriminator(inChannel, minChannel, maxChannel, nDownSampleLayer):
        return nn.Sequential(*(
            [DiscriminatorConvBlock(inChannel, minChannel, False, True)] +
            [DiscriminatorConvBlock(min(maxChannel, minChannel * 2 ** (i - 1)), 
                                    min(maxChannel, minChannel * 2 ** i), 
                                    True, True) 
             for i in range(1, nDownSampleLayer)] + 
            [DiscriminatorConvBlock(min(maxChannel, minChannel * 2 ** (nDownSampleLayer - 1)), 
                                    min(maxChannel, minChannel * 2 ** nDownSampleLayer)), 
             SpectralConv2d(min(maxChannel, minChannel * 2 ** nDownSampleLayer), 1, 4, 1, 2)]
        ))
    
    @staticmethod
    def GetOneDiscriminatorOuts(discriminator, x):
        outs = []
        for module in discriminator.children():
            outs.append(module(x))
            x = outs[-1]
        
        return outs