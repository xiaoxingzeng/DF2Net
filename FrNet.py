import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import math

class FrNet(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(FrNet, self).__init__()
        self.conv1_d_stage1 = nn.Conv2d(input_nc,8,3,1,1)
        self.relu1_d_stage1 = nn.LeakyReLU()
        self.conv1_c_stage1 = nn.Conv2d(input_nc*3,8,3,1,1)
        self.relu1_c_stage1 = nn.LeakyReLU()

        self.conv2_stage1 = nn.Conv2d(16,16,3,1,1)
        self.pool1_stage1 = nn.MaxPool2d(2,2)
        self.relu2_stage1 = nn.LeakyReLU()
        

        self.conv3_stage1 = nn.Conv2d(32,32,3,1,1)
        self.relu3_stage1 = nn.LeakyReLU()
        self.conv4_stage1 = nn.Conv2d(32,32,3,1,1)
        self.relu4_stage1 = nn.LeakyReLU()
        self.pool2_stage1 = nn.MaxPool2d(2,2)
        self.conv5_stage1 = nn.Conv2d(64,64,3,1,1)
        self.relu5_stage1 = nn.LeakyReLU()
        self.conv6_stage1 = nn.Conv2d(64,64,3,1,1)
        self.relu6_stage1 = nn.LeakyReLU()
        self.conv7_stage1 = nn.Conv2d(64,64,3,1,1)
        self.relu7_stage1 = nn.LeakyReLU()

        self.conv8_stage1 = nn.Conv2d(64,16,1)
        self.relu8_stage1 = nn.LeakyReLU()
        self.conv9_stage1 = nn.Conv2d(16,4,1)
        self.relu9_stage1 = nn.LeakyReLU()
        self.out_stage1   = nn.Conv2d(4,1,1)
        self.out_relu_stage1 = nn.Tanh()

        ###  stage 2 #########
        self.avgpool_img_4 = nn.AvgPool2d(4)
        self.conv1_d_stage2 = nn.Conv2d(input_nc,8,3,1,1)
        self.relu1_d_stage2 = nn.LeakyReLU()
        self.conv1_c_stage2 = nn.Conv2d(input_nc*3,8,3,1,1)
        self.relu1_c_stage2 = nn.LeakyReLU()

        
        self.conv2_stage2 = nn.Conv2d(16,16,3,1,1)
        self.relu2_stage2 = nn.LeakyReLU()
        self.up_stage2   = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv3_stage2 = nn.Conv2d(32,32,3,1,1)
        self.relu3_stage2 = nn.LeakyReLU()
        self.conv4_stage2 = nn.Conv2d(32,32,3,1,1)
        self.relu4_stage2 = nn.LeakyReLU()
        
        self.conv5_stage2 = nn.Conv2d(96,64,3,1,1)
        self.relu5_stage2 = nn.LeakyReLU()
        self.conv6_stage2 = nn.Conv2d(64,64,3,1,1)
        self.relu6_stage2 = nn.LeakyReLU()
        self.conv7_stage2 = nn.Conv2d(64,64,3,1,1)
        self.relu7_stage2 = nn.LeakyReLU()

        self.conv8_stage2 = nn.Conv2d(64,16,1)
        self.relu8_stage2 = nn.LeakyReLU()
        self.conv9_stage2 = nn.Conv2d(16,4,1)
        self.relu9_stage2 = nn.LeakyReLU()
        self.out_stage2   = nn.Conv2d(4,1,1)
        self.out_relu_stage2 = nn.Tanh()
#        
#        ###  stage 3 #########
        self.avgpool_img_2 = nn.AvgPool2d(2)  
        self.up_depth_stage3   = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv1_d_stage3 = nn.Conv2d(input_nc,8,3,1,1)
        self.relu1_d_stage3 = nn.LeakyReLU()
        self.conv1_c_stage3 = nn.Conv2d(input_nc*3,8,3,1,1)
        self.relu1_c_stage3 = nn.LeakyReLU()



        self.conv2_stage3 = nn.Conv2d(16,16,3,1,1)
        self.relu2_stage3 = nn.LeakyReLU()
        self.up_stage3   = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv3_stage3 = nn.Conv2d(32,32,3,1,1)
        self.relu3_stage3 = nn.LeakyReLU()
        self.conv4_stage3 = nn.Conv2d(32,32,3,1,1)
        self.relu4_stage3 = nn.LeakyReLU()
        
        self.conv5_stage3 = nn.Conv2d(64,64,3,1,1)
        self.relu5_stage3 = nn.LeakyReLU()
        self.conv6_stage3 = nn.Conv2d(64,64,3,1,1)
        self.relu6_stage3 = nn.LeakyReLU()
        self.conv7_stage3 = nn.Conv2d(64,64,3,1,1)
        self.relu7_stage3 = nn.LeakyReLU()

        self.conv8_stage3 = nn.Conv2d(64,16,1)
        self.relu8_stage3 = nn.LeakyReLU()
        self.conv9_stage3 = nn.Conv2d(16,4,1)
        self.relu9_stage3 = nn.LeakyReLU()
        self.out_stage3   = nn.Conv2d(4,1,1)
        self.out_relu_stage3 = nn.Tanh()
        
              
    def forward(self, depth,color):
        out1_d_stage1 = self.conv1_d_stage1(depth)
        out1_d_stage1 = self.relu1_d_stage1(out1_d_stage1)
        out1_c_stage1 = self.conv1_c_stage1(color)
        out1_c_stage1 = self.relu1_c_stage1(out1_c_stage1)
             
        conv1_stage1  = torch.cat([out1_d_stage1,out1_c_stage1],dim=1)
        out2_stage1   = self.conv2_stage1(conv1_stage1)
        out2_stage1   = self.relu2_stage1(out2_stage1)
        hyper1_stage1 = torch.cat([conv1_stage1,out2_stage1],dim=1)
             
        pool1_out_stage1  = self.pool1_stage1(hyper1_stage1)
        conv3_out_stage1  = self.conv3_stage1(pool1_out_stage1)
        conv3_out_stage1  = self.relu3_stage1(conv3_out_stage1)
        conv4_out_stage1  = self.conv4_stage1(conv3_out_stage1)
        conv4_out_stage1  = self.relu4_stage1(conv4_out_stage1)
             
        hyper2_stage1     = torch.cat([conv3_out_stage1,conv4_out_stage1],dim=1)
             
        pool2_out_stage1  = self.pool2_stage1(hyper2_stage1)
        conv5_out_stage1  = self.conv5_stage1(pool2_out_stage1)
        conv5_out_stage1  = self.relu5_stage1(conv5_out_stage1)
        conv6_out_stage1  = self.conv6_stage1(conv5_out_stage1)
        conv6_out_stage1  = self.relu6_stage1(conv6_out_stage1)
        conv7_out_stage1  = self.conv7_stage1(conv6_out_stage1)
        conv7_out_stage1  = self.relu7_stage1(conv7_out_stage1)
        conv8_out_stage1  = self.conv8_stage1(conv7_out_stage1)
        conv8_out_stage1  = self.relu8_stage1(conv8_out_stage1)
        conv9_out_stage1  = self.conv9_stage1(conv8_out_stage1)
        conv9_out_stage1  = self.relu9_stage1(conv9_out_stage1)
             
        out_final_stage1  = self.out_stage1(conv9_out_stage1)
        out_final_stage1  = self.out_relu_stage1(out_final_stage1)
         
#        ######  stage 2 ######
        color_128 = self.avgpool_img_4(color)
        depth_128 = self.avgpool_img_4(depth)
        out1_d_stage2 = self.conv1_d_stage2(out_final_stage1)
        out1_d_stage2 = self.relu1_d_stage2(out1_d_stage2)
        out1_c_stage2 = self.conv1_c_stage2(color_128)
        out1_c_stage2 = self.relu1_c_stage2(out1_c_stage2)
 
        
        conv1_stage2  = torch.cat([out1_d_stage2,out1_c_stage2],dim=1)
        
        out2_stage2   = self.conv2_stage2(conv1_stage2)
        out2_stage2   = self.relu2_stage2(out2_stage2)
        
        hyper1_stage2 = torch.cat([conv1_stage2,out2_stage2],dim=1)
 
        up1_out_stage2 = self.up_stage2(hyper1_stage2)
        up1_out_stage2 = torch.cat([up1_out_stage2,hyper2_stage1],dim=1)
        
        
        conv5_out_stage2  = self.conv5_stage2(up1_out_stage2)
        conv5_out_stage2  = self.relu5_stage2(conv5_out_stage2)
        conv6_out_stage2  = self.conv6_stage2(conv5_out_stage2)
        conv6_out_stage2  = self.relu6_stage2(conv6_out_stage2)
        conv7_out_stage2  = self.conv7_stage2(conv6_out_stage2)
        conv7_out_stage2  = self.relu7_stage2(conv7_out_stage2)
        conv8_out_stage2  = self.conv8_stage2(conv7_out_stage2)
        conv8_out_stage2  = self.relu8_stage2(conv8_out_stage2)
        conv9_out_stage2  = self.conv9_stage2(conv8_out_stage2)
        conv9_out_stage2  = self.relu9_stage2(conv9_out_stage2)
            
        out_final_stage2  = self.out_stage2(conv9_out_stage2)
        out_final_stage2  = self.out_relu_stage2(out_final_stage2)
#        
#        ######  stage 3 ######
        color_256 = self.avgpool_img_2(color)
        depth_256 = self.avgpool_img_2(depth)
        out1_d_stage3 = self.conv1_d_stage3(out_final_stage2)
        out1_d_stage3 = self.relu1_d_stage3(out1_d_stage3)
        
        out1_c_stage3 = self.conv1_c_stage3(color_256)
        out1_c_stage3 = self.relu1_c_stage3(out1_c_stage3)

        
        conv1_stage3  = torch.cat([out1_d_stage3,out1_c_stage3],dim=1)
        
        
        out2_stage3   = self.conv2_stage3(conv1_stage3)
        out2_stage3   = self.relu2_stage3(out2_stage3)
        hyper1_stage3 = torch.cat([conv1_stage3,out2_stage3],dim=1)
       
        
        up1_out_stage3 = self.up_stage3(hyper1_stage3)
        up1_out_stage3 = torch.cat([up1_out_stage3,hyper1_stage1],dim=1)
        
        
        conv5_out_stage3  = self.conv5_stage3(up1_out_stage3)
        conv5_out_stage3  = self.relu5_stage3(conv5_out_stage3)
        conv6_out_stage3  = self.conv6_stage3(conv5_out_stage3)
        conv6_out_stage3  = self.relu6_stage3(conv6_out_stage3)
        conv7_out_stage3  = self.conv7_stage3(conv6_out_stage3)
        conv7_out_stage3  = self.relu7_stage3(conv7_out_stage3)
        conv8_out_stage3  = self.conv8_stage3(conv7_out_stage3)
        conv8_out_stage3  = self.relu8_stage3(conv8_out_stage3)
        conv9_out_stage3  = self.conv9_stage3(conv8_out_stage3)
        conv9_out_stage3  = self.relu9_stage3(conv9_out_stage3)
           
        out_final_stage3  = self.out_stage3(conv9_out_stage3)
        out_final_stage3  = self.out_relu_stage3(out_final_stage3)
        
         
        return out_final_stage3