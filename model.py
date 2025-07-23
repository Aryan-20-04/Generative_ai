import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import multiprocessing

# ------------------------ #
# Command-line Arguments   #
# ------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
args = parser.parse_args()

# ------------------------ #
# Hyperparameters          #
# ------------------------ #
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
latent_dim = args.latent_dim
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

# ------------------------ #
# Data Preparation         #
# ------------------------ #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# ------------------------ #
# Model Definitions        #
# ------------------------ #
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

# ------------------------ #
# Training Setup           #
# ------------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

d_losses, g_losses = [], []

# ------------------------ #
# Training Loop            #
# ------------------------ #
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            curr_batch_size = real_images.size(0)

            real_labels = torch.ones(curr_batch_size, 1, device=device)
            fake_labels = torch.zeros(curr_batch_size, 1, device=device)

            # Train Discriminator
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(curr_batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            noise = torch.randn(curr_batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

        # Save model checkpoint
        torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_epoch{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_epoch{epoch+1}.pth"))

        # Show generated samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                noise = torch.randn(16, latent_dim, device=device)
                generated_images = generator(noise)
                grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
                plt.figure(figsize=(6, 6))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Epoch {epoch+1}")
                plt.axis('off')
                plt.show()

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("Training Complete.")
