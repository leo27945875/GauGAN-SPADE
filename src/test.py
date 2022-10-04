import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image

from model import Generator
from data  import Transforms, GetTestDataset
from utils import Denormalize, GPUToNumpy, LoadGenerator, GetColorMask, MaskToOnehot


def SynthesizeStyle(generator, onehot, style):
    generator.eval()
    with torch.no_grad():
        img = generator(style, onehot)[0]
    
    generator.train()
    return img


def GetRandomTestImages(dataset, testIndexList=[]):
    if not testIndexList:
        testIndexList = [random.randint(0, len(dataset) - 1) for _ in range(4)]
    
    imgs, masks, onehots, filenames = [], [], [], []
    for i in testIndexList:
        img, mask, onehot, filename = dataset.GetItemAndFilename(i)
        imgs     .append(img)
        masks    .append(mask)
        onehots  .append(onehot)
        filenames.append(filename)
    
    return imgs, masks, onehots, filenames


def TestGenerator(gen, images, masks, onehots, titles=[], device="cpu", saveFolder="", filename="test.png", isSave=False):
    nImg = len(images)
    gen.eval()
    with torch.no_grad():
        images   = torch.cat([img.unsqueeze(0) for img in images ], dim=0).to(device)
        onehots  = torch.cat([hot.unsqueeze(0) for hot in onehots], dim=0).to(device)
        fakeImgs = gen(images, onehots)[0]

        _, axes  = plt.subplots(nImg, 3, figsize=(10, 16))
        for i in range(nImg):
            realImg = np.clip(Denormalize(GPUToNumpy(images  [i])), 0, 1)
            fakeImg = np.clip(Denormalize(GPUToNumpy(fakeImgs[i])), 0, 1)
            mask = GetColorMask(masks[i].detach().cpu().numpy())
            axes[i, 0].set_title(titles[i])
            axes[i, 0].imshow(mask, cmap='gray')
            axes[i, 1].imshow(fakeImg)
            axes[i, 2].imshow(realImg)
        
    if isSave:
        plt.savefig(os.path.join(saveFolder, filename))
    else:
        plt.show()
    
    plt.clf()
    gen.train()


def LoadAndTestGenerator(generatorFilename, generatorFolder='./save', imageFolder='./archive/data', labelFolder='./archive/mask', saveFolder='./plot', epochs=100,
                         noiseDim=256, nClass=150, imageSize=128, device="cuda:0", imageChannel=3, maxChannel=1024, encoderMinChannel=64, encoderMaxChannel=512):
    device = torch.device(device)

    gen = Generator(noiseDim, nClass, imageSize, imageChannel, maxChannel, encoderMinChannel, encoderMaxChannel)
    gen = LoadGenerator(generatorFolder, generatorFilename, gen).to(device)

    dataset = GetTestDataset(imageFolder, labelFolder, imageSize, imageChannel)
    for epoch in range(epochs):
        testImgs, testMasks, testOnehot, testFilenames = GetRandomTestImages(dataset)
        TestGenerator(gen, testImgs, testMasks, testOnehot, testFilenames, device, saveFolder, f"Test_{epoch}.png", True)


def LoadAndGenerateImages(generatorFilename, generatorFolder='./save', imageFolder='./archive/data', labelFolder='./archive/mask', saveFolder='./plot',
                          noiseDim=256, nClass=150, imageSize=128, device="cuda:0", imageChannel=3, maxChannel=1024, encoderMinChannel=64, encoderMaxChannel=512):
    device = torch.device(device)

    gen = Generator(noiseDim, nClass, imageSize, imageChannel, maxChannel, encoderMinChannel, encoderMaxChannel)
    gen = LoadGenerator(generatorFolder, generatorFilename, gen).to(device)

    dataset = GetTestDataset(imageFolder, labelFolder, imageSize, imageChannel)
    with torch.no_grad():
        for i in range(len(dataset)):
            realImg, _, onehot = dataset[i]
            fakeImg = gen(realImg.to(device).unsqueeze(0), onehot.to(device).unsqueeze(0))[0][0]
            save_image(Denormalize(fakeImg), os.path.join(saveFolder, f"Generated_{i}.png"))


def LoadAndSynthesizeStyle(generatorFilename, styleFilename, labelFilename, saveFilename, generatorFolder='./save', styleFolder='./synthesis', labelFolder='./synthesis', saveFolder='./synthesis',
                           noiseDim=256, nClass=150, imageSize=128, device="cuda:0", imageChannel=3, maxChannel=1024, encoderMinChannel=64, encoderMaxChannel=512):
    device = torch.device(device)

    gen = Generator(noiseDim, nClass, imageSize, imageChannel, maxChannel, encoderMinChannel, encoderMaxChannel)
    gen = LoadGenerator(generatorFolder, generatorFilename, gen).to(device)

    style  = cv2.cvtColor(cv2.imread(os.path.join(styleFolder, styleFilename), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask   = cv2.imread(os.path.join(labelFolder, labelFilename), cv2.IMREAD_GRAYSCALE)
    concat = Transforms.GetTesting(imageSize, imageChannel)(image=style, mask=mask)

    synthesizedImg = SynthesizeStyle(gen, MaskToOnehot(concat['mask'].to(device).long()).unsqueeze(0), concat['image'].to(device).unsqueeze(0))[0]
    save_image(Denormalize(synthesizedImg), os.path.join(saveFolder, saveFilename))


if __name__ == '__main__':

    # LoadAndTestGenerator('Gen_300.pth')

    # LoadAndGenerateImages('Gen_300.pth')

    styleNum     = 3
    styleImage   = 'ADE_train_00011033'
    labelImage   = 'ADE_train_00005829'
    saveFilename = f'style_{styleImage}_label_{labelImage}'
    LoadAndSynthesizeStyle(
        'Gen_300.pth', 
        f'{styleNum}/{styleImage}.jpg', 
        f'{styleNum}/{labelImage}.png', 
        f'{styleNum}/{saveFilename}.jpg',
        './save/result10'
    )