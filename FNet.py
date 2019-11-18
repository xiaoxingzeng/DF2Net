import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import math

class FNet(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(FNet, self).__init__()
        self.conv1_d = nn.Conv2d(input_nc,8,3,1,1)
        self.relu1_d = nn.LeakyReLU()
        self.conv1_c = nn.Conv2d(input_nc*3,8,3,1,1)
        self.relu1_c = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.relu2 = nn.LeakyReLU()
        self.up1   = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)

        self.conv3 = nn.Conv2d(32,32,3,1,1)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32,32,3,1,1)
        self.relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(64,64,3,1,1)
        self.relu5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(64,64,3,1,1)
        self.relu6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(64,64,3,1,1)
        self.relu7 = nn.LeakyReLU()

        self.up2   = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=False)
        self.conv8 = nn.Conv2d(112,16,1)
        self.relu8 = nn.LeakyReLU()
        self.conv9 = nn.Conv2d(16,4,1)
        self.relu9 = nn.LeakyReLU()
        self.out   = nn.Conv2d(4,1,1)
        self.out_relu = nn.Tanh()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
                
                
    def forward(self, depth,color):
    	out1_d = self.conv1_d(depth)
    	out1_d = self.relu1_d(out1_d)
    	out1_c = self.conv1_c(color)
    	out1_c = self.relu1_c(out1_c)
    	
    	conv1  = torch.cat([out1_d,out1_c],dim=1)
    	out2   = self.conv2(conv1)
    	out2   = self.relu2(out2)
    	hyper1 = torch.cat([conv1,out2],dim=1)
    	
    	pool1_out  = self.pool1(hyper1)
    	conv3_out  = self.conv3(pool1_out)
    	conv3_out  = self.relu3(conv3_out)
    	conv4_out  = self.conv4(conv3_out)
    	conv4_out  = self.relu4(conv4_out)
    	
    	up1_out    = self.up1(conv4_out)
    	hyper2     = torch.cat([conv3_out,conv4_out],dim=1)
    	
    	pool2_out  = self.pool2(hyper2)
    	conv5_out  = self.conv5(pool2_out)
    	conv5_out  = self.relu5(conv5_out)
    	
    	conv6_out  = self.conv6(conv5_out)
    	conv6_out  = self.relu6(conv6_out)
    	
    	conv7_out  = self.conv7(conv6_out)
    	conv7_out  = self.relu7(conv7_out)
    	
    	up2_out    = self.up2(conv7_out)
    	
    	cat_f      = torch.cat([out2,up1_out],dim=1)
    	
    	cat_feature= torch.cat([cat_f,up2_out],dim=1)
    	
    	conv8_out  = self.conv8(cat_feature)
    	conv8_out  = self.relu8(conv8_out)
    	conv9_out  = self.conv9(conv8_out)
    	conv9_out  = self.relu9(conv9_out)
    	
    	out_final  = self.out(conv9_out)
    	out_final  = self.out_relu(out_final)

    	return out_final