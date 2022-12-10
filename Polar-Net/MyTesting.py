import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from utils.dataloader import test_dataset
import imageio
from lib.PolarNet_6x6PIE import Network

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size')
parser.add_argument('--pth_path', type=str, default='./Snapshot/Polar-Net/the_trained_model.pth')
parser.add_argument('--save_path', type=str, default='./result/Polar-Net/the_save_path_file/')
opt = parser.parse_args()

model = Network()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(opt.save_path, exist_ok=True)
image_root = 'testing_image_path/'
gt_root = 'ground_truth_path/'
test_loader = test_dataset(image_root, gt_root, opt.testsize)
for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    _, res = model(image)
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    name = name.split('image0')[0] + 'image0.png'
    print(name)
    imageio.imsave(opt.save_path + name, res)



