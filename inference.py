# import torch
# import yaml
# import os
# import numpy as np
# from tqdm import tqdm
# # import matplotlib.pyplot as plt
# from functions.process_data import *
# from main import dict2namespace
# from runners.image_editing import *
# from models.diffusion import Model
from colab_utils.utils import *
import cv2
from torchvision.utils import save_image
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LSUN', help='dataset name')
parser.add_argument('--category', type=str, default='church_outdoor', help='')
parser.add_argument('--data-name', type=str, default='lsun_church', help='')
parser.add_argument('--cfg-name', type=str, default='church.yml', help='')
parser.add_argument('--img-name', type=str, default='', help='')
parser.add_argument('--sample-step', type=int, default=3, help='Number of total repeats i.e. K')
parser.add_argument('--denoise', type=int, default=4, help='number of de-noising step i.e. N')
parser.add_argument('--t-noise', type=int, default=500, help='total noise level')
cfg = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
img_path = os.path.join('test_images', cfg['img-name'])
if not os.path.exists('results'):
    os.mkdir('results')
exit()
model, betas, num_timesteps, logvar = load_model(cfg['dataset'], cfg['category'], cfg['cfg-name'])
with torch.no_grad():
    print("Start sampling")
    mask_ = torch.zeros((3, 256, 256), dtype=torch.float)
    # mask_[:, :128] = 255
    # mask_ = torch.from_numpy(mask_).permute(2, 0, 1) / 255

    img_ = cv2.imread(img_path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img_, (256, 256))
    img_ = torch.from_numpy(img_).permute(2, 0, 1) / 255
    [mask, img] = mask_, img_
    mask = mask.to(device)
    save_image(mask, 'results/mask.png')
    # print(mask.size(), mask.max(), mask.min())
    img = img.to(device)
    save_image(img, 'results/img.png')
    img = img.unsqueeze(dim=0)
    img = img.repeat(cfg['denoise'], 1, 1, 1)
    x0 = img
    x0 = (x0 - 0.5) * 2.
    # plt.imshow(x0, title="Initial input")

    for it in range(cfg['sample_step']):
        e = torch.randn_like(x0)
        a = (1 - betas).cumprod(dim=0).to(device)
        x = x0 * a[cfg['t-noise'] - 1].sqrt() + e * (1.0 - a[cfg['t-noise'] - 1]).sqrt()
        # plt.imshow(x, title="Perturb with SDE")

        with tqdm(total=cfg['t-noise'], desc="Iteration {}".format(it)) as progress_bar:
            for i in reversed(range(cfg['t-noise'])):
                t = (torch.ones(cfg['denoise']) * i).to(device)
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
