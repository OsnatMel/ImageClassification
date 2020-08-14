
from imports import *
torch.manual_seed(SEED)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MobileNet_ConvAdd(nn.Module):
    def __init__(self):
        super(MobileNet_ConvAdd, self).__init__()

        self.model_M = models.mobilenet_v2(pretrained=True)
        #freezing parameters in order not to retrain
        for param in self.model_M.parameters():
            param.requires_grad = False

        self.MobileNet_ConvAdd_conv1 = nn.Sequential(
            
                                                    self.model_M.features[:], 
                                                    #Taking only the features - mobilenetv2 consists of features+activation function layer
                                                    #and classier as last layer. 
                                                    #In order to add more convolutional layers before a classifier layer - it was choosen 
                                                    #to take only the features and avoid the last two layers.
            
                                                    nn.Conv2d(1280, 1280, kernel_size=(2, 2), stride=(2, 2), bias=False),
                                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                                    nn.ReLU(),
                                                    nn.Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                                    nn.Flatten()
                                                    )
        #Separated to two Sequential for visual separation of the classifier and in case in the future an activation function 
        #will be inserted between as originally done in MobileNetV2
        
        self.MobileNet_ConvAdd_fc2 = nn.Sequential(
                                        nn.Dropout(0.2),
#                                         nn.BatchNorm1d(320),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),      

                                        nn.Linear(320,50)
                                        )
        self.MobileNet_ConvAdd_fc3 = nn.Sequential(
                                        nn.Linear(50,NUM_CLASSES),            
                                        nn.LogSoftmax(dim=1)
                                        )


    def forward(self, x):
        x = self.MobileNet_ConvAdd_conv1(x)
        x = self.MobileNet_ConvAdd_fc2(x)   
        MobileNet_ConvAdd_fc2_x = x #saving the 50 features for clustering purpose
        x = self.MobileNet_ConvAdd_fc3(x)
        return x,MobileNet_ConvAdd_fc2_x

######################################################################################################################################
    
def create_model():

    model = MobileNet_ConvAdd()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-5)
      
    model.to(DEVICE)
    summary(model, input_size=(3, 224, 224))

    ########################
    #documentation
    f = open(TB_DIR+"/model_description.txt","w+")
    f.write("~~~~~~~~~~~~\r\n")
    f.write("-- "+COMMENT+"\r\n")
    f.write("-- Model layers \r\n")
    for p in str(model).split('\n'):
        f.write("   "+str(p)+"\r\n")

    f.write("-- "+LOSS+"\r\n")
    f.write("-- "+OPTIM+" optimizer \r\n")
    for var_name in optimizer.state_dict():
        v = var_name, ":", optimizer.state_dict()[var_name]
        f.write("   "+str(v)+"\r\n")

    f.close()
    ########################

    return model, criterion , optimizer
