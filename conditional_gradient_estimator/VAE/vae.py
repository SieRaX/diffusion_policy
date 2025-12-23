import torch
import torch.nn as nn
import torch.nn.functional as F

# # ---------------------------
# #  VAE model
# # ---------------------------
# class ConvEncoder(nn.Module):
#     def __init__(self, in_channels=6, latent_dim=128):
#         super().__init__()
#         # Input: [B, 6, 84, 84]
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # -> [B, 32, 42, 42]
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          # -> [B, 64, 21, 21]
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)         # -> [B, 128, 21, 21]

#         self.latent_dim = latent_dim
#         self.feature_dim = 128 * 21 * 21

#         self.fc_mu     = nn.Linear(self.feature_dim, latent_dim)
#         self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

#     def forward(self, x):
#         # x: [B, 6, 84, 84]
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.reshape(x.size(0), -1)  # [B, feature_dim]

#         mu     = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar


# class ConvDecoder(nn.Module):
#     def __init__(self, out_channels=6, latent_dim=128):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.feature_dim = 128 * 21 * 21

#         self.fc = nn.Linear(latent_dim, self.feature_dim)

#         # Mirror of encoder
#         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)  # 21 -> 21
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 21 -> 42
#         self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 42 -> 84

#     def forward(self, z):
#         # z: [B, latent_dim]
#         x = self.fc(z)
#         x = x.reshape(x.size(0), 128, 21, 21)
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = torch.sigmoid(self.deconv3(x))  # output in [0,1] if inputs are normalized that way
#         return x  # [B, 6, 84, 84]


# class VAE(nn.Module):
#     def __init__(self, in_channels=6, latent_dim=128):
#         super().__init__()
#         self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
#         self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decoder(z)
#         return recon_x, mu, logvar


# # ---------------------------
# #  VAE loss
# # ---------------------------
# def vae_loss(recon_x, x, mu, logvar, beta=1.0):
#     """
#     recon_x, x: [B, 6, 84, 84]
#     Reconstruction loss + KL divergence
#     """
#     # Use MSE or BCE; here MSE
#     rec_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

#     # KL divergence between N(mu, sigma) and N(0, I)
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

#     loss = rec_loss + beta * kld
#     return loss, rec_loss, kld

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=6, latent_dim=128, input_size=84):
        super().__init__()
        # Input: [B, in_channels, input_size, input_size]
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # -> [B, 32, input_size/2, input_size/2]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          # -> [B, 64, input_size/4, input_size/4]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)         # -> [B, 128, input_size/4, input_size/4]

        self.latent_dim = latent_dim
        # Calculate feature dimensions dynamically based on input_size
        # After conv1: input_size/2, after conv2: input_size/4, after conv3: input_size/4
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        self.feature_dim = 128 * self.feature_h * self.feature_w

        self.fc_mu     = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        # x: [B, in_channels, input_size, input_size]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # [B, feature_dim]

        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=6, latent_dim=128, input_size=84):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        # Calculate feature dimensions to match encoder
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        self.feature_dim = 128 * self.feature_h * self.feature_w

        self.fc = nn.Linear(latent_dim, self.feature_dim)

        # Mirror of encoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)  # input_size/4 -> input_size/4
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # input_size/4 -> input_size/2
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # input_size/2 -> input_size

    def forward(self, z):
        # z: [B, latent_dim]
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.feature_h, self.feature_w)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # output in [0,1] if inputs are normalized that way
        return x  # [B, out_channels, input_size, input_size]


class VAE(nn.Module):
    def __init__(self, in_channels=6, latent_dim=7, input_size=84):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim, input_size=input_size)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim, input_size=input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# ---------------------------
#  VAE loss
# ---------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    recon_x, x: [B, C, H, W] where C=6 (stacked images), H and W are input dimensions
    Reconstruction loss + KL divergence
    """
    # Use MSE or BCE; here MSE
    rec_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence between N(mu, sigma) and N(0, I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    loss = rec_loss + beta * kld
    return loss, rec_loss, kld