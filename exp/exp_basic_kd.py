import os

import torch

from models import MTS_31, MTS_31F


class Exp_Basic:
    """Minimal experiment base for S14 KD and causal image stages."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "MTS_31": MTS_31,
            "MTS_31F": MTS_31F,
        }
        unsupported = set((args.model, args.teacher_model)) - set(self.model_dict)
        if unsupported:
            raise ValueError(
                "FACTS_NEW only supports MTS_31/MTS_31F, got "
                + ", ".join(sorted(unsupported))
            )
        self.device = self._acquire_device()
        self.model, self.model_teacher = self._build_model()
        self.model.to(self.device)
        if self.model_teacher is not None:
            self.model_teacher.to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            visible = self.args.devices if self.args.use_multi_gpu else str(self.args.gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
            return device
        if self.args.use_gpu and self.args.gpu_type == "mps":
            print("Use GPU: mps")
            return torch.device("mps")
        print("Use CPU")
        return torch.device("cpu")
