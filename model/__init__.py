from .NoiseUnet import NoiseUnet
from .NoiseTransformer import NoiseTransformer
from .SVDNoiseUnet import SVDNoiseUnet


model_dict = {
      'unet': NoiseUnet,
      'vit': NoiseTransformer,
      'svd_unet': SVDNoiseUnet,
      'svd_unet+unet': [SVDNoiseUnet, NoiseUnet]
}