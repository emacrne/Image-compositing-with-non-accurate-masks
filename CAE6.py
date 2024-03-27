import torch
import torch.nn as nn


class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        k = 5

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, k, stride=1, padding=2),  
            torch.nn.BatchNorm2d(8),                               
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(8, 16, k, stride=1, padding=2),  
            torch.nn.BatchNorm2d(16),                               
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(16, 32, k, stride=1, padding=2),  
            torch.nn.BatchNorm2d(32),                               
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(32, 64, k, stride=1, padding=1),  
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1), 
            torch.nn.Conv2d(64, 128,k, stride=1, padding=1),  
            torch.nn.BatchNorm2d(128), 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),  
            torch.nn.Conv2d(128, 256, k, stride=1, padding=1),  
            torch.nn.BatchNorm2d(256), 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256*2, 128, k, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, k, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 32, k, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 16, k, stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, k, stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 3, k, stride=1, padding=1),  
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        #print(x.shape)
        coded = self.encode(x)
        #print(coded.shape)
        decoded = self.decode(coded)
        #print(decoded.shape)
        #x = decoded.view(-1, 3, 224, 224)       
        return decoded
    

class CustomFusionModel(nn.Module):
    def __init__(self, autoencoder1, autoencoder2):
        super(CustomFusionModel, self).__init__()
        self.autoencoder1 = autoencoder1
        self.autoencoder2 = autoencoder2
        
    
    def forward(self, image1, image2, mask, train):

        if train:
            self.autoencoder1.train()
            self.autoencoder2.train()
        else:
            self.autoencoder1.eval()
            self.autoencoder2.eval()
    
        encoded1 = self.autoencoder1.encode(torch.cat((image1, mask), dim=1))
        encoded2 = self.autoencoder2.encode(torch.cat((image2, 1 - mask), dim=1))
        
        
        fused_features = torch.cat((encoded1, encoded2), dim=1)

        reconstructed_image = self.autoencoder1.decode(fused_features)
        
        return reconstructed_image