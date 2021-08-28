#%% 
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import *
import pickle as pkl
import pandas as pd
import random
import cv2

def get_test_input():
    img = cv2.imread("cocoimages/dog-cycle-car.png")
    img = cv2.resize(img,(608,608))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

#%%
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):       
            
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                
    
            elif module_type == "route":
                
                # concat layers
                layers = module["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:    # 1 item in layer                 
                    x = outputs[i + (layers[0])]
    
                else:   # more than 1 item in layer 
                    if len(layers) == 4:       # 4 items in layer                                      
                        if (layers[1]) > 0:
                            layers[1] = layers[1] - i

                        if (layers[2]) > 0:
                            layers[2] = layers[2] - i

                        if (layers[3]) > 0:
                            layers[3] = layers[3] - i

                        map1 = outputs[i + layers[0]]
                        map2 = outputs[i + layers[1]]
                        map3 = outputs[i + layers[2]]
                        map4 = outputs[i + layers[3]]
                        x = torch.cat((map1, map2, map3, map4), 1)

                    else:           # 2 items in layer                
                        if (layers[1]) > 0:
                            layers[1] = layers[1] - i
        
                        map1 = outputs[i + layers[0]]
                        map2 = outputs[i + layers[1]]
                        x = torch.cat((map1, map2), 1)
                
                
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
                
            
            outputs[i] = x
        
        
        return detections


    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            
            if i > 104:
                break
            
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


#%%
model = Darknet("cfg/yolov4.cfg")
model.module_list[114].conv_114 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.load_weights("csdarknet53-omega_final.weights")

# %%

inp = get_test_input()
pred = model(inp, False)
print(pred)

print(pred.shape)

pred_fin = write_results(pred, 0.5, 80, nms_conf = 0.4)
print(pred_fin.shape)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate(batch):
    return tuple(zip(*batch))
    # images = []
    # bboxes = []
    # for img, box in batch:
    #     images.append([img])
    #     bboxes.append([box])
    # images = np.concatenate(images, axis=0)
    # images = images.transpose(0, 3, 1, 2)
    # images = torch.from_numpy(images).div(255.0)
    # bboxes = np.concatenate(bboxes, axis=0)
    # bboxes = torch.from_numpy(bboxes)
    # return images, bboxes
#%%
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import cv2
import numpy as np
import torch
import albumentations as A

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
# classes = load_classes("data/coco.names")


path2data="/../../../root/labs/COCOdata/val2017"
path2json="/../../../root/labs/COCOdata/annotations/instances_val2017.json"

coco_train = dset.CocoDetection(root = path2data, annFile = path2json)

# get all 
all_img = []
all_boxes = []
all_cat = []
for i, (img, anno) in enumerate(coco_train):
    if i == 100:
        break
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_img.append(img)

    boxes_one_pic = []
    cat_one_pic = []
    for each_box in anno:        
        boxes_one_pic.append(each_box['bbox'])
        cat_index = each_box['category_id']
        cat_one_pic.append(cat_index)
    all_boxes.append(boxes_one_pic)
    all_cat.append(cat_one_pic)

#print(len(all_img))
# print(len(all_boxes[6]))
# print(all_cat)


#%%

transform = A.Compose([
    A.Resize(608,608),
    A.RandomCrop(608,608),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)]
    , bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


all_transformed_images = torch.Tensor()
all_transform_labels = []
for i in range(len(all_img)):
    img = all_img[i]
    box = all_boxes[i]
    class_labels = all_cat[i]
    transformed = transform(image=img, bboxes=box,class_labels=class_labels)
    transformed_image = torch.from_numpy(np.transpose(transformed['image'],(2,0,1))).reshape(1,3,608,608)
    all_transformed_images = torch.cat((all_transformed_images,transformed_image), dim=0)

    this_img_label = []
    transformed_bboxes = torch.Tensor(transformed['bboxes']).reshape(-1,4)
    transformed_class_labels = torch.Tensor(transformed['class_labels']).reshape(-1,1)
    this_img_label.append(transformed_bboxes)
    this_img_label.append(transformed_class_labels)
    all_transform_labels.append(this_img_label)

print(all_transformed_images.shape)
print(all_transform_labels)
all_transform_labels = torch.Tensor(all_transform_labels)

stim_loader = torch.utils.data.DataLoader((all_transformed_images,all_transform_labels), batch_size=5)
for image,labels in stim_loader:
    print(image.shape)
    print(labels)
    break

# dataset = ds = TensorDataset(all_transformed_images,all_transform_labels)




# print(transformed_image)
# print(transformed_bboxes)

# from torch.utils.data import DataLoader
# train_loader = DataLoader(coco_train, batch_size=2, shuffle=True,
#                               num_workers=1, pin_memory=True, drop_last=True, collate_fn=collate)                            

# %%
