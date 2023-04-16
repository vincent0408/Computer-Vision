import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        layers = []
        in_channels = 3
        for i in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if i == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_channels, i, kernel_size=3, padding=1),
                           nn.BatchNorm2d(i),
                           nn.ReLU(inplace=True)])
                in_channels = i
        self.nnet = nn.Sequential(*layers,
        nn.AvgPool2d(kernel_size=1, stride=1),
        nn.Flatten(),
        nn.Linear(512, 10)
        )

    def forward(self, x):
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        output = self.nnet(x)
        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
        self.resnet.maxpool = Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    # model = ResNet18()
    # print(model)
    print(ResNet18())
    print(MyNet())
