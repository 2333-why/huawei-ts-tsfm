import os

import torch

from models import MTS_31F


class Exp_Basic:
    """Minimal experiment base for the privileged Stanford teacher."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {"MTS_31F": MTS_31F}
        if args.model not in self.model_dict:
            raise ValueError(f"FACTS_NEW teacher only supports MTS_31F, got {args.model}")
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
            return device
        if self.args.use_gpu and self.args.gpu_type == "mps":
            print("Use GPU: mps")
            return torch.device("mps")
        print("Use CPU")
        return torch.device("cpu")

