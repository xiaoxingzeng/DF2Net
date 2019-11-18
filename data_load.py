import os, sys, shutil
import random as rd
import struct as st
import dlib
from PIL import Image
import numpy as np
from scipy import misc
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from numpy import *

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


###    reference to SFSNet : https://github.com/senguptaumd/SfSNet/tree/master/functions/create_mask_fiducial.m  
def crop_face_mask(img_path):
    raw_im = cv2.imread(img_path)
    raw_gray = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)
    dets = face_cascade.detectMultiScale(raw_gray, 1.3, 5)
    if not isinstance(dets,tuple):
       for (x,y,w,h) in dets:
           x_r = int(np.max((0,min(raw_im.shape[0],x-w*0.2))))
           y_r = int(np.max((0,min(raw_im.shape[1],y-h*0.2))))
           w_r = int(np.max((0,min(raw_im.shape[0],w*1.5))))
           h_r = int(np.max((0,min(raw_im.shape[0],h*1.5))))
           roi_color = raw_im[ y_r:h_r+y_r,x_r:x_r+w_r]
   
       crop_im = cv2.resize(roi_color,(512,512))
       im = cv2.resize(roi_color,(224,224))
       rects_pre = detector(im,1)
       
       if len(rects_pre)==1:
          out = get_landmarks(im)
          out = out.transpose(1,0)  # out shape (2,68)
          border_fid = out[:,0:17]
          face_fid = out[:,17:]
          c1 = [border_fid[0,0],face_fid[1,2]]
          c2=[border_fid[0,16], face_fid[1,7]]
          eye_distance=np.linalg.norm(face_fid[:,22]-face_fid[:,25]) 
          c3=face_fid[:,0]
          c3[1]=c3[1]-1*eye_distance
          c3 = c3.transpose(1,0)
          c4=face_fid[:,9]
          c4[1]=c4[1]-1*eye_distance
          c4=c4.transpose(1,0)
          all_points = np.zeros((2,21),np.uint8)
          all_points[:,0:17] = border_fid
          all_points[:,17]=c2
          all_points[:,18]=c4
          all_points[:,19]=c3
          all_points[:,20]=c1
          all_points_mask = all_points.copy()
          all_points[0,0:8] = all_points[0,0:8]+4
          all_points[0,9:17] = all_points[0,9:17]-8     
          all_points[:,17] = all_points[:,17] -8
          all_points[:,18] = all_points[:,18] +4
          all_points[:,19] = all_points[:,19] +4
          all_points[:,20] = all_points[:,20] +4
          print all_points
          mask = np.zeros((224,224,1))
          mask_mask = np.zeros((224,224,1))
          all_points = all_points.transpose(1,0)
          all_points_mask = all_points_mask.transpose(1,0)
          cv2.fillPoly(mask, np.array([all_points], dtype=np.int32), (255, 255, 255))
          cv2.fillPoly(mask_mask, np.array([all_points_mask], dtype=np.int32), (255, 255, 255))
          mask = cv2.resize(mask,(512,512))
          mask = mask/255.0
          mask_mask = cv2.resize(mask_mask,(512,512))
          mask_mask = mask_mask/255.0
          crop_im = crop_im*mask_mask[:,:,np.newaxis]
          cv2.imwrite(img_path.replace('.png','_crop.png'),crop_im)
          cv2.imwrite(img_path.replace('.png','_mask.png'),mask*255)
       else:
          pass

def load_imgs(img_dir, image_list_file):
    imgs = list()
    with open(image_list_file, 'r') as imf:
        for line in imf:
            record = line.strip().split()
            record_path = record[0]
            
            img_path = record_path
            img_path = os.path.join(img_dir,record_path)
            imgs.append((img_path,record_path))
    return imgs


class ImageDataset(data.Dataset):
    def __init__(self, img_dir, image_list_file, transform=None):
        self.imgs = load_imgs(img_dir, image_list_file)
        self.transform = transform

    def __getitem__(self, index):
        path,img_name = self.imgs[index]
        print 'path',path
        crop_face_mask(path)
        img = Image.open(path.replace('.png','_crop.png')).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img,img_name
    
    def __len__(self):
        return len(self.imgs)


class CaffeCrop(object):
    #This class take the same behavior as sensenet
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 512
        final_width = final_height = final_size
        
        res_img = img.resize( (final_width, final_height) )
        return res_img
