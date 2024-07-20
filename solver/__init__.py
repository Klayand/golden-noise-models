from ESVD_NoiseSolver import SVD_Embedding_Solver
from EUnet_NoiseSolver import Embedding_Solver
from Unet_NoiseSolver import Unet_Solver
from VIT_NoiseSolver import Vit_Solver
from SVD_NoiseSolver import SVD_Solver


solver_dict = {
      'unet': Unet_Solver,
      'vit': Vit_Solver,
      'e_unet': Embedding_Solver,
      'svd_unet': SVD_Solver,
      'svd_unet+unet': SVD_Embedding_Solver
}