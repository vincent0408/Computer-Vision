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
        self.nnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # initialize second set of CONV => RELU => POOL layers
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            # initialize first (and only) set of FC => RELU layers
            nn.Linear(in_features=1250, out_features=500),
            nn.ReLU(),
            # initialize our softmax classifier
            nn.Linear(in_features=500, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        # pass the input through our first set of CONV => RELU =>
		# POOL layers
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        output = self.nnet(x)
        # return the output predictions
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
    model = ResNet18()
    print(model)
