import torch

from model import Generator, Discriminator, Vgg19
from loss  import GeneratorLoss, DiscriminatorLoss
from data  import GetImageLoader, GetTestDataset
from utils import LinearDecayScheduler, SaveCheckPoint, SeedEverything, DetachAll, PrintTrainMessage, SaveCheckPoint
from test  import GetRandomTestImages, TestGenerator


def Train(imageFolder, labelFolder, saveFolder, nWorker=0, imageSize=256, imageChannel=3, noiseDim=256, nClass=150, nDiscriminator=3, nDiscriminatorDownSampleLayer=3,
          epochs=300, batchSize=32, batchStep=1, lr=1e-4, ttur=4, adamBetas=(0., 0.9), wGANLoss=1., wFeatureMatch=10., wPerceptual=10., wKLDivergence=0.001, 
          isTest=True, isSaveGeneratedImage=True, isSaveCheckpoint=True, testIndexList=[0, 1, 2, 3], startSaveEpoch=170, startLRDecayEpoch=200, seed=42):
    # Fix random seed:
    SeedEverything(seed)

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloader & dataset:
    batchStepSize = batchSize // batchStep
    dataloader, dataset = GetImageLoader(imageFolder, labelFolder, imageSize, imageChannel, batchStepSize, nWorker)
    maxNumBatch = len(dataset) // batchSize if len(dataset) % batchSize == 0 else len(dataset) // batchSize + 1

    # Models:
    gen = Generator(noiseDim, nClass, imageSize, imageChannel).to(device)
    dis = Discriminator(imageChannel + nClass, nDownSampleLayer=nDiscriminatorDownSampleLayer, nDiscriminator=nDiscriminator).to(device)

    # Loss functions:
    genLossFunc = GeneratorLoss(Vgg19(), wGANLoss, wFeatureMatch, wPerceptual, wKLDivergence).to(device)
    disLossFunc = DiscriminatorLoss().to(device)

    # Optimizers:
    genOpt = torch.optim.Adam(gen.parameters(), lr       , adamBetas)
    disOpt = torch.optim.Adam(dis.parameters(), lr * ttur, adamBetas)

    # LR schedulers:
    genScheduler = LinearDecayScheduler(genOpt, startLRDecayEpoch, epochs)
    disScheduler = LinearDecayScheduler(disOpt, startLRDecayEpoch, epochs)

    # Mixed-precision training scalers:
    genScaler = torch.cuda.amp.GradScaler()
    disScaler = torch.cuda.amp.GradScaler()

    # Testing cases:
    testImg, testMasks, testOnehot, testFilenames = GetRandomTestImages(
        GetTestDataset(imageFolder, labelFolder, imageSize, imageChannel), 
        testIndexList
    )

    # Training loop:
    for epoch in range(1, epochs + 1):
        for batch, (realImg, _, onehot) in enumerate(dataloader, 1):
            # Preprocess variales:
            realImg, onehot, isUpdate = realImg.to(device), onehot.to(device), (batch % batchStep == 0)

            # Train discriminator:
            gen.requires_grad_(False)
            dis.requires_grad_(True)
            with torch.cuda.amp.autocast():
                fakeImg = gen(realImg, onehot)[0].detach()
                fakeOutsList = dis(fakeImg, onehot)
                realOutsList = dis(realImg, onehot)
                lossDis = disLossFunc(realOutsList, fakeOutsList) / batchStep
            
            disScaler.scale(lossDis).backward()
            if isUpdate:
                disScaler.step(disOpt)
                disScaler.update()
                disOpt.zero_grad()

            # Train generator:
            gen.requires_grad_(True)
            dis.requires_grad_(False)
            with torch.cuda.amp.autocast():
                fakeImg, mu, logVar = gen(realImg, onehot)
                fakeOutsList = dis(fakeImg, onehot)
                realOutsList = dis(realImg, onehot)
                lossGen = genLossFunc(realImg, DetachAll(realOutsList), fakeImg, fakeOutsList, mu, logVar) / batchStep
            
            genScaler.scale(lossGen).backward()
            if isUpdate:
                genScaler.step(genOpt)
                genScaler.update()
                genOpt.zero_grad()

            # Print training message:
            if isUpdate:
                PrintTrainMessage(epoch, batch // batchStep, epochs, maxNumBatch, lossDis.item(), lossGen.item())
        
        print('')

        # Test generator:
        if isTest:
            TestGenerator(gen, testImg, testMasks, testOnehot, testFilenames, device, saveFolder, f"Test_{epoch}.png", isSaveGeneratedImage)
        
        # Save checkpoints:
        if isSaveCheckpoint and epoch >= startSaveEpoch:
            SaveCheckPoint(saveFolder, f"Gen_{epoch}.pth", epoch, gen, genOpt, None, genScaler)
            SaveCheckPoint(saveFolder, f"Dis_{epoch}.pth", epoch, dis, disOpt, None, disScaler)
        
        # Update schedulers:
        disScheduler.Update(epoch)
        genScheduler.Update(epoch)


if __name__ == '__main__':

    Train('./archive/data', './archive/mask', './save', 6, 128, batchStep=4, batchSize=64, nDiscriminatorDownSampleLayer=3, nDiscriminator=3)

