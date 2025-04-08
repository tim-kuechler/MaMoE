import torch as th
import torch.nn as nn
from diffusers import AutoencoderKL


class VaeWrapper(nn.Module):
    """A Wrapper around an AutoencoderKL to make encoding into and decoding from latent space
    easier. Especially already applies scaling with the autoencoder scaling_factor.

    forward(...) function defaults to encode(...).
    """

    def __init__(self, vae: AutoencoderKL) -> None:
        """Inits a VaeWrapper.

        :param vae: Autoencoder to wrap around.
        """
        super().__init__()
        self.vae = vae

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward method of the VaeWrapper. Defaults to img encoding.

        :param x: RGB image in shape (B, 3, H, W)
        :return: Latent image of shape (B, 4, H/F, W/F)
        """
        return self.encode(x)

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Encodes an RGB image to a latent image. Also handles scaling of the latent image after
        encoding.

        :param x: RGB image in shape (B, 3, H, W)
        :return: Latent image of shape (B, 4, H/F, W/F)
        """
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, x: th.Tensor) -> th.Tensor:
        """Decodes a latent image to an RGB image. Also handles scaling of the latent image before
        decoding and clamping to [-1, 1] after decoding.

        :param x: Latent image in shape (B, 4, H/F, W/F)
        :return: Latent image of shape (B, 3, H, W)
        """
        return self.vae.decode(x / self.vae.config.scaling_factor).sample.clamp(-1, 1)


def init_vae(model_name: str) -> VaeWrapper:
    """Initializes the VAE part of SD.

    The loaded model will be returned in eval mode without any trainable parameters. The returned
    class is a wrapper around the original VAE. It includes a forward defaulting to encoding as
    well a separate encode and decode method. In these methods also the scaling of the latent image
    is included.

    :param model_name: The name of the pretrained base model.
    :return: VaeWrapper to use for training.
    """
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    vae.requires_grad_(False)
    vae = VaeWrapper(vae)
    vae.eval()

    return vae
