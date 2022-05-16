import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt

# pytorch helpfully makes it easy to download datasets, e.g. the common CIFAR-10 https://www.kaggle.com/c/cifar-10
#root = './'
#train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
#test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
#print(len(train_data), len(test_data))

import os
from PIL import Image
#img_dir = '/Users/fanyang/Dropbox/art_data/kelly'
img_dir = '/Users/fanyang/Dropbox/art_data/calder'
train_data = [(Image.open(os.path.join(img_dir, path)).resize((32, 32)).convert('RGB'), -1) for path in os.listdir(img_dir)]
print(len(train_data))

# make deterministic
from mingpt.utils import set_seed
set_seed(42)


# TODO(fyang): this can be improved by a learning approach, e.g. autoencoder.
# get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()
print(px.size())
print(len(set([tuple(x.numpy()) for x in px])))

# run kmeans to get our codebook
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # compute the loss as the l2 distance between elements and assigned clusters
        loss = torch.sqrt(((x - torch.stack([c[i] for i in a], dim=0))**2).sum(-1)).mean(0)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('loss %0.4f, done step %d/%d, re-initialized %d dead clusters' % (loss, i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c

ncluster = 512
with torch.no_grad():
    C = kmeans(px, ncluster, niter=8)

print(C.size())


# encode the training examples with our codebook to visualize how much we've lost in the discretization
n_samples = 16
ncol = 8
nrow = n_samples // ncol + 1
plt.figure(figsize=(20, 10))
for i in range(n_samples):
    
    # encode and decode random data
    x, y = train_data[np.random.randint(0, len(train_data))]
    xpt = torch.from_numpy(np.array(x)).float().view(32*32, 3)
    ix = ((xpt[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1) # cluster assignments for each pixel
    
    # these images should look normal ideally
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(C[ix].view(32, 32, 3).numpy().astype(np.uint8))
    plt.axis('off')
    plt.savefig('./coded.png')
