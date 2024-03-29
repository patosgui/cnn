from torch import tensor
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split, Subset

import torch.nn as nn


def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks // 2)
    if act:
        res = nn.Sequential(res, nn.ReLU())
    return res


def apply_single_kernel_to_img(first_image):
    def apply_kernel(img, row, col, kernel):
        return (img[row - 1 : row + 2, col - 1 : col + 2] * kernel).sum()

    top_edge = tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).float()

    rng = range(1, 27)
    top_edges = tensor(
        [[apply_kernel(first_image, i, j, top_edge) for j in rng] for i in rng]
    )

    image = Image.fromarray(top_edges.numpy())
    image.show()


TRAIN_SPLIT = 0.95
VAL_SPLIT = 1 - TRAIN_SPLIT

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

numTrainSamples = int(len(training_data) * TRAIN_SPLIT)
numValSamples = int(len(training_data) * VAL_SPLIT)

limited_data = Subset(
    training_data, torch.arange(numTrainSamples + numValSamples)
)

(trainData, valData) = random_split(
    limited_data,
    [numTrainSamples, numValSamples],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 64

trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=batch_size)
valDataLoader = DataLoader(valData, batch_size=batch_size)
testDataLoader = DataLoader(test_data, batch_size=batch_size)

first_image = next(iter(trainDataLoader))[0][0][0]
transform = ToPILImage()
image = transform(first_image)
image.show()

# first_image = test_ds["image"][0]
simple_net = nn.Sequential(
    conv(1, 4),  # 14x14
    conv(4, 8),  # 7x7
    conv(8, 16),  # 4x4
    conv(16, 32),  # 2x2
    conv(32, 10, act=False),  # 1x1
    nn.Flatten(),
)

summary(simple_net, input_size=(1, 28, 28), batch_size=1)

# initialize the Adam optimizer with the loss function
opt = optim.Adam(simple_net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 12

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

trainSteps = len(trainDataLoader.dataset) // batch_size
valSteps = len(valDataLoader.dataset) // batch_size


# Copied from https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# loop over our epochs
for e in range(0, n_epochs):
    # set the model in training mode
    simple_net.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for x, y in trainDataLoader:
        # perform a forward pass and calculate the training loss
        pred = simple_net(x)
        loss = loss_fn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        ## switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            simple_net.eval()
            # loop over the validation set
            for x, y in valDataLoader:
                # make the predictions and calculate the validation loss
                pred = simple_net(x)
                totalValLoss += loss_fn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (
                    (pred.argmax(1) == y).type(torch.float).sum().item()
                )

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
    print(
        "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect
        )
    )
    print(
        "Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect
        )
    )


print("RESULT ")
print(first_image.shape)
first_image = first_image.unsqueeze(0)
print(first_image.shape)
print(simple_net(first_image))
