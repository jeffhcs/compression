import torch
import torch.nn as nn

class SoftRound(nn.Module):
    def __init__(self):
        super().__init__()

    def round(self, x):
        return x - torch.sin(2 * torch.pi * x) / (3 * torch.pi)

    def forward(self, x):
        x = self.round(x * 255)
        x = self.round(x)
        x = self.round(x)
        return x

class FGAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(FGAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Maintains 92x84
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 46x42
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Maintains 46x42
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 23x21
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Maintains 23x21
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 11x10
            nn.Flatten(),
            nn.Linear(64 * 11 * 10, latent_dim),
            nn.ReLU()
            # nn.Sigmoid(),
            # SoftRound()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 11 * 10),
            nn.ReLU(),
            nn.Unflatten(1, (64, 11, 10)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Outputs 23x21
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=1),  # Outputs 46x42
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Outputs 92x84
            nn.Sigmoid()  # Ensures output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class BGAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(BGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Maintains 218x178
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 109x89
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Maintains 109x89
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 54x44
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Maintains 54x44
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 27x22
            nn.Flatten(),
            nn.Linear(64 * 27 * 22, latent_dim),
            nn.ReLU()
            # nn.Sigmoid(),
            # SoftRound()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 27 * 22),
            nn.ReLU(),
            nn.Unflatten(1, (64, 27, 22)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Outputs 54x44
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Outputs 109x89
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=0),  # Outputs 218x178
            nn.Sigmoid()  # Ensures output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TwoResAutoEncoder(nn.Module):
    def __init__(self, high_latent_dim, low_latent_dim):
        super(TwoResAutoEncoder, self).__init__()
        self.fg_ae = FGAutoencoder(high_latent_dim)
        self.bg_ae = BGAutoencoder(low_latent_dim)

    def forward(self, image, face):
        fg_output = self.fg_ae(face)
        bg_output = self.bg_ae(image)
        
        return fg_output, bg_output