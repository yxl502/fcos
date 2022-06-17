import torch.nn as nn
import torch.nn.functional as F
import math

class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x
        """
        C3 torch.Size([8, 512, 104, 136])
        C4 torch.Size([8, 1024, 52, 68])
        C5 torch.Size([8, 2048, 26, 34])
        """
        P5 = self.prj_5(C5)
        # print('P5', P5.size())  # torch.Size([8, 256, 26, 34])
        P4 = self.prj_4(C4)
        # print('P4', P4.size())  # torch.Size([8, 256, 52, 68])
        P3 = self.prj_3(C3)
        # print('P3', P3.size())  # torch.Size([8, 256, 104, 136])
        
        P4 = P4 + self.upsamplelike([P5,C4])
        # print('P41', P4.size())  # torch.Size([8, 256, 52, 68])
        P3 = P3 + self.upsamplelike([P4,C3])
        # print('P31', P3.size())  # torch.Size([8, 256, 104, 136])

        P3 = self.conv_3(P3)
        # print('P32', P3.size())  # torch.Size([8, 256, 104, 136])
        P4 = self.conv_4(P4)
        # print('P42', P4.size())  # torch.Size([8, 256, 52, 68])
        P5 = self.conv_5(P5)
        # print('P52', P5.size())  # torch.Size([8, 256, 26, 34])
        
        P5 = P5 if self.use_p5 else C5
        # print('P53', P5.size())  # torch.Size([8, 256, 26, 34])
        P6 = self.conv_out6(P5)
        # print('P63', P6.size())   # torch.Size([8, 256, 13, 17])
        P7 = self.conv_out7(F.relu(P6))
        # print('P73', P7.size())   # torch.Size([8, 256, 7, 9])
        return [P3,P4,P5,P6,P7]


if __name__ == '__main__':
    """
        C3 torch.Size([8, 512, 104, 136])
        C4 torch.Size([8, 1024, 52, 68])
        C5 torch.Size([8, 2048, 26, 34])

    """
    import torch
    C3 = torch.rand(size=([8, 512, 104, 136]))
    C4 = torch.rand(size=([8, 1024, 52, 68]))
    C5 = torch.rand(size=([8, 2048, 26, 34]))
    [P3, P4, P5, P6, P7] = FPN()([C3, C4, C5])
    print(P3.size())
    print(P4.size())
    print(P5.size())
    print(P6.size())
    print(P7.size())
    """
    torch.Size([8, 256, 104, 136])
    torch.Size([8, 256, 52, 68])
    torch.Size([8, 256, 26, 34])
    torch.Size([8, 256, 13, 17])
    torch.Size([8, 256, 7, 9])
    """
