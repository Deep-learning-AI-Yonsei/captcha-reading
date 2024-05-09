import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import GPT2LMHeadModel, GPT2Config

class Encoder(nn.Module): # ViT Encoder
    def __init__(self, embed_size, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear = nn.Linear(self.vit.config.hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_size = embed_size

    def forward(self, images):
        features = self.vit(pixel_values=images).last_hidden_state
        features = features.mean(dim=1)
        features = self.dropout(self.linear(features))
        return features

class Decoder(nn.Module): # GPT-2 Decoder
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_rate=0.1, gpt2_model_name='gpt2-medium'):
        super(Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.linear = nn.Linear(embed_size, self.gpt2.config.n_embd)
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_size = embed_size

    def forward(self, features, captions, lengths):
        transformed_features = self.linear(features).unsqueeze(1)
        transformed_features = self.dropout(transformed_features)

        embeddings = self.gpt2.transformer.wte(captions)
        inputs_embeds = torch.cat([transformed_features, embeddings], dim=1)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds).logits
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False).data
        return packed_outputs 

    def sample(self, features, max_length=20):
        transformed_features = self.linear(features).unsqueeze(1)
        transformed_features = self.dropout(transformed_features)

        generated = transformed_features
        result_ids = []

        for i in range(max_length):
            outputs = self.gpt2(inputs_embeds=generated)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            result_ids.append(next_token)
            next_token_embedding = self.gpt2.transformer.wte(next_token)
            generated = torch.cat([generated, next_token_embedding], dim=1)

        return torch.cat(result_ids, dim=1)