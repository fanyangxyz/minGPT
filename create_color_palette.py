import os
import colorsys

from PIL import Image
import numpy as np
import torch


def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # compute the loss as the l2 distance between elements and assigned
        # clusters
        loss = torch.sqrt(
            ((x - torch.stack([c[i] for i in a], dim=0))**2).sum(-1)).mean(0)
        # move each codebook element to be the mean of the pixels that assigned
        # to it
        c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print(
            'loss %0.4f, done step %d/%d, re-initialized %d dead clusters' %
            (loss, i + 1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
    return c


def rgb_to_hsv(image):
    hei, wid, _ = image.shape
    image = np.reshape(image, (-1, 3)) / 255.
    res = []
    for im in image:
        h, s, v = colorsys.rgb_to_hsv(*im)
        res.append((h, s, v))
    return np.reshape(np.array(res), (hei, wid, 3))


def hsv_to_rgb(C):
    res = []
    for c in C:
        r, g, b = colorsys.hsv_to_rgb(*c)
        res.append([r, g, b])
    return np.array(res) * 255


def main():
    from mingpt.utils import set_seed
    set_seed(33)

    array = np.array(Image.open('samples/sampled_small_model.png'))
    array = array[:, :, :3]
    thm = 85
    bhm = 75
    lwm = 180
    rwm = 140
    cropped = array[thm:-bhm, lwm:-rwm, :]
    os.makedirs('images/', exist_ok=True)
    os.makedirs('palettes/', exist_ok=True)
    os.makedirs('palettes_hsv/', exist_ok=True)
    size = 160
    for i in range(32):
        r, c = divmod(i, 8)
        image = cropped[r * size:(r + 1) * size, c * size:(c + 1) * size, :]
        Image.fromarray(image).save(f'images/image_{i}.png')
        nc = 4
        image = rgb_to_hsv(image)
        C = kmeans(torch.Tensor(image).view(-1, 3), nc**2, niter=20).numpy()
        C = hsv_to_rgb(C)
        C = np.rint(C).astype(np.uint8)
        C = sorted(C, key=lambda x: x[0])
        # Image.fromarray(np.reshape(C, (nc, nc, 3))).save(f'palettes/palette_{i}.png')
        Image.fromarray(np.reshape(C, (nc, nc, 3))).save(
            f'palettes_hsv/palette_{i}.png')
        # break


if __name__ == '__main__':
    main()
