import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
import torchvision
from kornia_moons.viz import draw_LAF_matches
import torchvision.transforms.functional as F
import torchvision.transforms as T
from algorithm_creation import *

matcher = KF.LoFTR(pretrained='outdoor')

main('/content/drive/MyDrive/img_match_sentinel/jp2_img01.png',
     '/content/drive/MyDrive/img_match_sentinel/jp2_imgNN.png',
      matcher,
      10980,
      10980,
      20,
      1200,
      1200)
