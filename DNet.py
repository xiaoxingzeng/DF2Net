import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from FNet import FNet
from FrNet import FrNet
import scipy.io as sio
def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%srelu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%sleakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%sconv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%stconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%sbn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%sdropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class DNet(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(DNet, self).__init__()

    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    print 'name',name
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

    
    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    
    #import pdb; pdb.set_trace()
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    
    d_inc = nf*2

    dlayer1 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    
    
    add_relu_1 = nn.ReLU(inplace=True)
    add_transpose_conv_1 = nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False)
    add_transpose_bn_1   = nn.BatchNorm2d(64)
    add_relu_2 = nn.ReLU(inplace=True)
    add_transpose_conv_2 = nn.ConvTranspose2d(64, 32, 3, 1, 1, bias=False)
    add_transpose_bn_2   = nn.BatchNorm2d(32)
    add_relu_3 = nn.ReLU(inplace=True)
    add_transpose_conv_3 = nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False)


    # dlayer0.add_module('%sconv' % name, nn.Conv2d(64, 64, 3, 1, 1, bias=False))
    # #dlayer0.add_module('%sbn' % name, nn.BatchNorm2d(64))
    # dlayer0.add_module('%srelu' % name, nn.ReLU(inplace=True))
    # dlayer0.add_module('%sconv_1' % name, nn.Conv2d(64, 32, 3, 1, 1, bias=False))
    # #dlayer0.add_module('%sbn_1' % name, nn.BatchNorm2d(32))
    # dlayer0.add_module('%srelu_2' % name, nn.ReLU(inplace=True))
    # dlayer0.add_module('%sconv_2' % name, nn.Conv2d(32, 1, 3, 1, 1, bias=False))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    

    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1
    # self.dlayer0 = dlayer0
    self.add_relu_1=add_relu_1
    self.add_transpose_conv_1=add_transpose_conv_1
    self.add_transpose_bn_1 = add_transpose_bn_1
    self.add_relu_2 = add_relu_2
    self.add_transpose_conv_2 = add_transpose_conv_2
    self.add_transpose_bn_2 = add_transpose_bn_2
    self.add_relu_3 = add_relu_3
    self.add_transpose_conv_3 = add_transpose_conv_3
    #self.UNet_sfs = UNet_sfs(1,1,64)
    self.FNet = FNet(1,1,64)
    #self.Hyper_Net_Cascade_Part = Hyper_Net_Cascade_Part(1,1,64)
    self.FrNet = FrNet(1,1,64)
    
    self.avgpool_img_4 = nn.AvgPool2d(4)
    self.avgpool_img_2 = nn.AvgPool2d(2)
  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    
    
   
   
   
   
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    dout_add1 = self.add_relu_1(dout1)
    dout_add2 = self.add_transpose_conv_1(dout_add1)
    dout_add3 = self.add_transpose_bn_1(dout_add2)
    dout_add4 = self.add_relu_2(dout_add3)
    dout_add5 = self.add_transpose_conv_2(dout_add4)
    dout_add6 = self.add_transpose_bn_2(dout_add5)
    dout_add7 = self.add_relu_3(dout_add6)
    dout_add8 = self.add_transpose_conv_3(dout_add7)

    refine_depth = self.FNet(dout_add8,x)
    #depth_512= self.Hyper_Net_Cascade_Part(refine_depth,x)
    depth_512 = self.FrNet(refine_depth,x)
    #####   resolution 512  #######
    

    return dout_add8+depth_512*3
    
    