import os
import cv2
import glob
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

from utils import Denormalize, MaskToOnehot, CountLabel


class ReplayBuffer:
    def __init__(self, n=1000, imageSize=128):
        self.buffer  = np.zeros([n, 4, imageSize, imageSize])
        self.pointer = 0
        self.isFull  = False
    
    def AddBatch(self, batchImage, batchMask):
        batch = torch.cat([batchImage.cpu(), batchMask.cpu().unsqueeze(1)], dim=1).numpy()
        start, end, n = self.pointer, self.pointer + batch.shape[0], self.buffer.shape[0]
        if end <= n:
            self.buffer[start: end] = batch
        else:
            end -= n
            self.buffer[start:] = batch[:(n - start)]
            self.buffer[:end  ] = batch[(n - start):]
            self.isFull = True
        
        self.pointer = end
    
    def SampleBatch(self, batchSize, device="cpu"):
        index = random.sample(range(self.buffer.shape[0]), k=batchSize)
        batchImage  = torch.FloatTensor([self.buffer[i][:3] for i in index])
        batchOnehot = torch.cat([MaskToOnehot(torch.LongTensor(self.buffer[i][-1])).unsqueeze(0) for i in index])
        return batchImage.to(device), batchOnehot.to(device)
    
    def AddToFull(self, generator, dataloader, device="cpu"):
        with torch.no_grad():
            while not self.isFull:
                for realImg, mask, onehot in dataloader:
                    fakeImg = generator(realImg.to(device), onehot.to(device))[0]
                    self.AddBatch(fakeImg, mask)
                    if self.isFull: break


class Transforms:
    @staticmethod
    def GetTraining(imageSize, imageChannel=3):
        return A.Compose([
            A.RandomCrop(imageSize, imageSize),
            A.HorizontalFlip(p=0.5),
            A.Normalize([0.5] * imageChannel, [0.5] * imageChannel, 255),
            ToTensorV2()
        ])
    
    @staticmethod
    def GetTesting(imageSize, imageChannel=3):
        return A.Compose([
            A.CenterCrop(imageSize, imageSize),
            A.Normalize([0.5] * imageChannel, [0.5] * imageChannel, 255),
            ToTensorV2()
        ])


class ImageLabelDataset(Dataset):
    def __init__(self, imageFolder, labelFolder, transform=ToTensorV2()):
        self.imageFolder = imageFolder
        self.labelFolder = labelFolder
        self.transform   = transform
        self.CheckToTensor()

        self.imagePaths = glob.glob(os.path.join(imageFolder, '*'))
        self.labelPaths = glob.glob(os.path.join(labelFolder, '*'))
        self.CheckImageLabelNumber()
    
    def __getitem__(self, i):
        image       = cv2.cvtColor(cv2.imread(self.imagePaths[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask        = cv2.imread(self.labelPaths[i], cv2.IMREAD_GRAYSCALE)
        concat      = self.transform(image=image, mask=mask)
        image, mask = concat['image'], concat['mask'].long()
        return image, mask, MaskToOnehot(mask)
    
    def __len__(self):
        return len(self.imagePaths)

    def GetItemAndFilename(self, i):
        imgPath, labelPath = self.imagePaths[i], self.labelPaths[i]
        image       = cv2.cvtColor(cv2.imread(imgPath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask        = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)
        concat      = self.transform(image=image, mask=mask)
        image, mask = concat['image'], concat['mask'].long()
        return image, mask, MaskToOnehot(mask), os.path.splitext(os.path.basename(imgPath))[0]

    def CheckToTensor(self):
        transform = self.transform
        if type(transform) is not ToTensorV2:
            if type(transform) is A.Compose:
                for t in transform.transforms:
                    if isinstance(t, ToTensorV2): break
                else:
                    raise TypeError("Need [ToTensorV2] in transform.")
            
            else:
                raise TypeError("Need [ToTensorV2] in transform.")
    
    def CheckImageLabelNumber(self):
        if len(self.imagePaths) != len(self.labelPaths):
            raise ValueError("[len(self.imagePaths)] and [len(self.labelPaths)] must be equal.")
    
    def GetImageFromName(self, name):
        for i, path in enumerate(self.imagePaths):
            if name in path:
                return self[i]


class ResizeToMinSize:
    def __init__(self, size, interpolation=cv2.INTER_CUBIC, isForceSize=True):
        self.size          = size
        self.interpolation = interpolation
        self.isForceSize   = isForceSize
    
    def __call__(self, image):
        size = self.size
        h, w = image.shape[:2]

        if self.isForceSize or (h < size or w < size):
            scaleH, scaleW = size / h, size / w
            scale = scaleH if scaleH >= scaleW else scaleW
            image = cv2.resize(image, (int(np.ceil(w * scale)), int(np.ceil(h * scale))), interpolation=self.interpolation)
        
        return image

def GetImageLoader(imageFolder, labelFolder, imageSize, imageChannel, batchSize, nWorker):
    transform  = Transforms.GetTraining(imageSize, imageChannel)
    dataset    = ImageLabelDataset(imageFolder, labelFolder, transform)
    dataLoader = DataLoader(dataset, 
                            batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=nWorker)
    return dataLoader, dataset


def GetTestDataset(imageFolder, labelFolder, imageSize, imageChannel):
    return ImageLabelDataset(imageFolder, labelFolder, Transforms.GetTesting(imageSize, imageChannel))


def PreResizeImagesAndMasks(imageSrcFolder='./archive/origin/image', imageDstFolder='./archive/data', maskSrcFolder='./archive/origin/annotation', maskDstFolder='./archive/mask',
                            minSize=128, imageInterpolation=cv2.INTER_CUBIC, maskInterpolation=cv2.INTER_NEAREST):
    imgPaths, maskPaths = glob.glob(os.path.join(imageSrcFolder, '*')), glob.glob(os.path.join(maskSrcFolder, '*'))
    for i, (imgPath, maskPath) in enumerate(zip(imgPaths, maskPaths), 1):
        print(f'\r{i}', end='')

        img = cv2.imread(imgPath)
        img = ResizeToMinSize(minSize, imageInterpolation)(img)
        cv2.imwrite(os.path.join(imageDstFolder, os.path.basename(imgPath)), img)

        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        mask = ResizeToMinSize(minSize, maskInterpolation)(mask)
        mask[mask >= 9] -= 1
        cv2.imwrite(os.path.join(maskDstFolder, os.path.basename(maskPath)), mask)


def SaveTestImage(imageFolder, imageSize=128, saveFolder='./save'):
    trans = Transforms.GetTesting(imageSize)
    for path in glob.glob(os.path.join(imageFolder, '*')):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = trans(image=img)['image']
        img = Denormalize(img).numpy().transpose(1, 2, 0) * 255

        nameExt   = os.path.basename(path)
        name, ext = os.path.splitext(nameExt)
        savePath  = os.path.join(saveFolder, name + f'_Test.{ext}')
        cv2.imwrite(savePath, img)



if __name__ == '__main__':
    from pprint import pprint
    
    # PreResizeImagesAndMasks(minSize=int(128 * 1.5))

    # transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.Normalize([0.5] * 3, [0.5] * 3),
    #     ToTensorV2()
    # ])
    # dataset = ImageLabelDataset('./archive/data', './archive/mask', transform)

    # for name in ['ADE_train_00005829', 'ADE_train_00005854', 'ADE_train_00005831', 'ADE_train_00005882']:
    #     img, mask, onehot = dataset.GetImageFromName(name)
    #     pprint(CountLabel(mask, list(range(150))))

    # print('-' * 100)
    # pprint(mask)
    # print('-' * 100)
    # pprint(onehot[:, 0, 0])
    # print('-' * 100)
    # print(img.shape, mask.shape)
    # print('-' * 100)
    # print(onehot.dtype)

    folder = './synthesis/2'
    SaveTestImage(folder, 128, folder)
