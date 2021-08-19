import torch
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functions.process_data import *
from main import dict2namespace
from runners.image_editing import *
from models.diffusion import Model
from colab_utils.utils import *
import cv2
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "LSUN"
category = "church_outdoor"
data_name = "lsun_church"
sample_step = 3
total_noise_levels = 500
n = 4
cfg_name = 'church.yml'

model, betas, num_timesteps, logvar = load_model(dataset, category, cfg_name)
with torch.no_grad():
    print("Start sampling")
    mask_ = torch.zeros((3, 512, 512), dtype=torch.float)
    # mask_[:, :128] = 255
    # mask_ = torch.from_numpy(mask_).permute(2, 0, 1) / 255
    # mask_ = cv2.circle(mask_, (35, 60), 30, (255,255,255), -1)

    # img_ = cv2.imread('/content/2.jpg') # Bedroom
    # img_ = cv2.imread('/content/Episcopal_Church_1600x900.jpg') # church

    img_ = cv2.imread('test_images/sagrada_1.jpg')  # church
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img_, (512, 512))
    img_ = torch.from_numpy(img_).permute(2, 0, 1) / 255
    # _, img = torch.load("colab_demo/{}.pth".format(name))
    [mask, img] = mask_, img_
    mask = mask.to(device)
    save_image(mask, 'results/mask.png')
    # print(mask.size(), mask.max(), mask.min())
    img = img.to(device)
    save_image(img, 'results/img.png')
    img = img.unsqueeze(dim=0)
    img = img.repeat(n, 1, 1, 1)
    x0 = img
    x0 = (x0 - 0.5) * 2.
    # plt.imshow(x0, title="Initial input")

    for it in range(sample_step):
        e = torch.randn_like(x0)
        a = (1 - betas).cumprod(dim=0).to(device)
        x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
        # plt.imshow(x, title="Perturb with SDE")

        with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
            for i in reversed(range(total_noise_levels)):
                t = (torch.ones(n) * i).to(device)
                x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                logvar=logvar,
                                                                betas=betas)
                x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                # added intermediate step vis
                # if (i - 99) % 100 == 0:
                    # plt.imshow(x, title="Iteration {}, t={}".format(it, i))
                progress_bar.update(1)

        x0[:, (mask != 1.)] = x[:, (mask != 1.)]
        save_image(x, 'results/out_{}.png'.format(it))


