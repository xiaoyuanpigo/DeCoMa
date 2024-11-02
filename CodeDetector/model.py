import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward(self, input_embeddings, input_mask, decoder_input_ids):
        outputs = self.model(inputs_embeds=input_embeddings, attention_mask=input_mask,
                             decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        logits = outputs["logits"][:, 0, :]
        return logits

    def generate(self, input_ids, max_new_tokens, return_dict_in_generate, temperature):
        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens,
                                      return_dict_in_generate=return_dict_in_generate, temperature=temperature)
        return outputs
