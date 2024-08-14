from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import (
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig
)


class LoraCodeLlama(nn.Module):
    
    def __init__(self, model):
        super(LoraCodeLlama, self, ).__init__()
        
        self.model = model
        self.config = self.model.config
        #self.model.resize_token_embeddings(self.model.base_model.model.model.embed_tokens.weight.size(0) + 8) #Assertion `srcIndex < srcSelectDimSize`
    def forward(self, input_ids, attention_mask, labels, past_key_values = None):

        if attention_mask is None:
            # auto-regressive generation
            return self.model.forward(input_ids=input_ids,past_key_values = past_key_values)
        else:
            
            output = self.model(
                input_ids=input_ids,
                attention_mask = attention_mask,
                labels= labels,
                return_dict = True
            )
            loss = output.loss
            return {"loss":loss,"hidden_states":output.hidden_states}
        