import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy


class Seq2Seq(nn.Module):
    def __init__(self, model, config):
        super(Seq2Seq, self).__init__()
        self.model = model
        self.config = config

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             decoder_attention_mask=decoder_attention_mask)
        return outputs

    def generate(self, input_ids, attention_mask, use_cache, num_beams, early_stopping, max_length):
        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            num_beams=num_beams,
            early_stopping=early_stopping,
            max_length=max_length
        )
