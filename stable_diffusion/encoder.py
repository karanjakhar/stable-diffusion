import torch 
from torch import nn 
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    

    def __init__() -> None:
        super().__init__(
            # (Batch_Size, channel, height, width) -> (Batch_Size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_Size, 128, height, width) -> (Batch_Size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, height, width) -> (Batch_Size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, height, width) -> (Batch_Size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, height/2, width/2) -> (Batch_Size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256), 

            # (Batch_Size, 256, height/2, width/2) -> (Batch_Size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, height/2, width/2) -> (Batch_Size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            
            # (Batch_Size, 256, height/4, width/4) -> (Batch_Size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, height/4, width/4) -> (Batch_Size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),


            # (Batch_Size, 512, height/4, width/4) -> (Batch_Size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 512, height/8, width/8)
            nn.SiLU(),

            # (Batch_Size, 512, height/8, width/8) -> (Batch_Size, 8, height / 8, width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, height / 8, width / 8) -> (Batch_Size, 8, height / 8, width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),


        )


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, channel, Height, Width)
        # noise: (Batch_Size, out_channels, height/8, width/8)


        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0,1,0,1))

            x = module(x)

        # (batch_size, 8, height / 8, width / 8) -> two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        #(batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        # (batch_size, 4, height/8, width/8)
        variance = log_variance.exp()
        
        # (batch_size, 4, height/8, width/8)
        std_dev = variance.sqrt()

        x = mean + std_dev * noise

        # scale the output by a constant
        x *= 0.18215

        return x
