import argparse
import os,sys,shutil
import time
import struct as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import dlib
import cv2
import numpy as np
import scipy.io as sio

from data_load import ImageDataset, CaffeCrop
import scipy.io as sio
import numpy as np
import scipy
import scipy.io as sio
import cv2
from DNet import DNet
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Face Reconstruction')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--model_dir','-m', default='./model', type=str)




def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def face_reconstruction(resume):
    global args
    args = parser.parse_args()
    
    # load data and prepare dataset
    demo_list_file = 'img_list.txt'
    img_dir = './img/'
    out_dir = './output/'

    caffe_crop = CaffeCrop('test')
    demo_dataset =  ImageDataset(img_dir, demo_list_file,
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    demo_loader = torch.utils.data.DataLoader(
        demo_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
   

    model = None
    model = DNet(3,1,64)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()



    assert(os.path.isfile(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True


    for i, (input,img_name) in enumerate(demo_loader):
        
        img_name = str(img_name)
        img_name = img_name[2:-2]
        out_mat_name = out_dir+img_name.replace('.png','.mat')

        input_var = torch.autograd.Variable(input, volatile=True)
        input_var = input_var*2 -1
        start = time.clock()
        output_shape = model(input_var)
        elapsed = (time.clock() - start)
        output_shape = ((output_shape+1.0)/2.0)*255
        output_shape = output_shape.cpu().data.numpy()
    
        if os.path.exists(img_dir+img_name.replace('.png','_mask.png')):
        
           mask_n = cv2.imread(img_dir+img_name.replace('.png','_mask.png'))
           mask_n = mask_n.sum(2)
           mask_n[mask_n!=0]=1
           feat_batch  = output_shape.shape[0]
           out_im = output_shape[0,0,:,:]
           out_im = out_im*mask_n[:,:]
           sio.savemat(out_mat_name,{'depth_mat':out_im})
           print img_name
            



if __name__ == '__main__':
    
    model_path = './model/DF2Net.pth.tar'
    start = time.clock()
    face_reconstruction(model_path)
    elapsed = (time.clock() - start)
    print 'Time used: ',elapsed
