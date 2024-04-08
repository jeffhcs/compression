import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7)  
        
        # Decoder
        self.convtranspose1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=7)
        self.convtranspose2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtranspose3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = F.relu(self.convtranspose1(x))
        x = F.relu(self.convtranspose2(x))
        x = torch.sigmoid(self.convtranspose3(x))  # Using sigmoid to output values between 0 and 1
        
        return x

def main():
    model = CNNAutoencoder()
    random_input = torch.rand(10, 1, 100, 100)
    output = model(random_input)

    # output = model(random_input)
    print(output)
    

if __name__ == "__main__":
    main()
