from pyaet.resire_numpy.RESIRE_Reconstructor import RESIRE_Reconstructor as NumpyRESIREReconstructor


class RESIRE_Reconstructor(NumpyRESIREReconstructor):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.dtype_torch = "float64"
        self.gpu_grad_device = "cuda"
