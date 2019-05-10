import torch
import os
import numpy as np

# BATCH_SIZE should divide train size and validation size due to groups in convolution
TEST = False
LOG_INTERVAL = 100  # 100

# TODO change this before cluster
IN_CHANNELS_RECONSTRUCTION = 2 # 2 was what the basis was trained with
BATCH_SIZE = 100  # 100

PLOT_IMAGES = False

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    i = np.argmax(memory_available)
    print("device with max memory: "+str(i))
    return i


idx = get_freer_gpu()
device_str = "cuda:" + str(idx)
DEVICE = torch.device(device_str)
OVERFIT_SUBSET = False
