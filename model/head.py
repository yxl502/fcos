
import torch.nn as nn
import torch
import math

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg
        
        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):
            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits,cnt_logits,reg_preds


if __name__ == '__main__':
    from config import DefaultConfig as config
    print(config.fpn_out_channels)  # 256
    print(config.class_num)  # 3
    print(config.use_GN_head)  # True
    print(config.cnt_on_reg)   # True
    print(config.prior)   # 0.01

    # all_P = [tensor1 ,tensor2, tensor3, tensor4, tensor5]
    # torch.Size([8, 256, 104, 136])
    # torch.Size([8, 256, 52, 68])
    # torch.Size([8, 256, 26, 34])
    # torch.Size([8, 256, 13, 17])
    # torch.Size([8, 256, 7, 9])
    all_P = [torch.rand(size=(8, 256, 104, 136)),
             torch.rand(size=(8, 256, 52, 68)),
             torch.rand(size=(8, 256, 26, 34)),
             torch.rand(size=(8, 256, 13, 17)),
             torch.rand(size=(8, 256, 7, 9))]
    cls_logits, cnt_logits, reg_preds = \
        ClsCntRegHead(config.fpn_out_channels, config.class_num,
                  config.use_GN_head, config.cnt_on_reg, config.prior)(all_P)
    print(len(cls_logits), len(cnt_logits), len(reg_preds))
    print(cls_logits[0].size(),
          cnt_logits[0].size(),
          reg_preds[0].size())
    print(cls_logits[1].size(),
          cnt_logits[1].size(),
          reg_preds[1].size())
    print(cls_logits[2].size(),
          cnt_logits[2].size(),
          reg_preds[2].size())
    print(cls_logits[3].size(),
          cnt_logits[3].size(),
          reg_preds[3].size())
    print(cls_logits[4].size(),
          cnt_logits[4].size(),
          reg_preds[4].size())

    """
torch.Size([8, 3, 104, 136]) torch.Size([8, 1, 104, 136]) torch.Size([8, 4, 104, 136])
torch.Size([8, 3, 52, 68]) torch.Size([8, 1, 52, 68]) torch.Size([8, 4, 52, 68])
torch.Size([8, 3, 26, 34]) torch.Size([8, 1, 26, 34]) torch.Size([8, 4, 26, 34])
torch.Size([8, 3, 13, 17]) torch.Size([8, 1, 13, 17]) torch.Size([8, 4, 13, 17])
torch.Size([8, 3, 7, 9]) torch.Size([8, 1, 7, 9]) torch.Size([8, 4, 7, 9])

    """

        
