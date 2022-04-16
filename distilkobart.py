import os
import warnings
from typing import List, Union, Optional, Callable

import torch
from torch import nn

from transformers import BartForConditionalGeneration
from transformers.modeling_utils import save_pretrained


LAYERS_TO_COPY = {
    # maps num layers in teacher -> num_layers in student -> which teacher layers to copy
    # 6: KoBart
    6: { # maps num layers in student -> which teacher layers to copy
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 5],
        3: [0, 2, 5],
        4: [0, 1, 3, 5],
        6: list(range(6))
    },
}


class DistilKoBart(nn.Module):
    def __init__(self, teacher, e=None, d=None):
        super(DistilKoBart, self).__init__()

        self.teacher = teacher
        init_kwargs = teacher.config.to_diff_dict()

        teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"encoder_layers": e, "decoder_layers": d})

        # Copy weights
        student_cfg = teacher.config_class(**init_kwargs)
        self.config = student_cfg
        self.model = BartForConditionalGeneration(config=student_cfg)
        self.model.load_state_dict(self.teacher.state_dict(), strict=False)

        e_layers_to_copy: List[int] = self.pick_layers_to_copy(e, teacher_e)
        d_layers_to_copy: List[int] = self.pick_layers_to_copy(d, teacher_d)

        self.copy_layers(
            self.teacher.model.encoder.layers,
            self.model.model.encoder.layers,
            e_layers_to_copy
        )
        self.copy_layers(
            self.teacher.model.decoder.layers,
            self.model.model.decoder.layers,
            d_layers_to_copy
        )
        self.teacher = None


    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs
        )


    def copy_layers(self, src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
        layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
        assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
        dest_layers.load_state_dict(layers_to_copy.state_dict())


    def pick_layers_to_copy(self, n_student, n_teacher):
        try:
            val = LAYERS_TO_COPY[n_teacher][n_student]
            return val
        except KeyError:
            if n_student != n_teacher:
                warnings.warn(
                    f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
                )
            return list(range(n_student))
    

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)
