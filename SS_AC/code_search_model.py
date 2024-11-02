import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

    def get_t5_vec(self, source_ids, source_embeddings, source_mask):
        if source_ids != None and source_embeddings == None and source_mask == None:
            attention_mask = source_ids.ne(0)
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                   labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        elif source_embeddings != None and source_mask != None:
            outputs = self.encoder(inputs_embeds=source_embeddings, attention_mask=source_mask,
                                   labels=source_ids, decoder_attention_mask=source_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(2)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_inputs=None, source_embeddings=None, source_mask=None):
        outputs = self.get_t5_vec(source_inputs, source_embeddings, source_mask)
        return torch.nn.functional.normalize(outputs, p=2, dim=1)

    def forward_emb(self, source_embeddings, source_mask):
        outputs = self.get_t5_vec(source_embeddings=source_embeddings, source_mask=source_mask)[0]
        return outputs
