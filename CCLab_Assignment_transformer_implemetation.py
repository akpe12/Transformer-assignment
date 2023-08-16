import torch
import torch.nn as nn
import math

'''
2022.08.21 CCLab Study 과제 by sungminlee

Transformer Implementation

Transformer architecture의 핵심은
1. multi-head attention
2. Masking

이고, 구현에서도 손이 제일 가는 부분입니다.

따라서 진행하셔야할 부분은,

1. Positional Encoding 구현
2. Multi-Head Attention 구현
(추가적으로, 가능하다면, 구현된 transformer를 이용해 Translation, Summarization 등 Task에 학습을 진행해보는 것입니다.)


'''

# 구현 1
class PosEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.positional_encoder = None
    
    def pe(self, pos, i):
        frequency = 10000
        
        sine_pos = math.sin(pos / math.pow(frequency, (2 * i) / self.config.transformer_hidden_size))
        cosine_pos = math.cos(pos / math.pow(frequency, (2 * i) / self.config.transformer_hidden_size))
        
        return (sine_pos, cosine_pos)
        
    def forward(self, seq_len):
        self.positional_encoder = torch.zeros((seq_len, self.config.transformer_hidden_size))
        # self.positional_encoder = torch.zeros((seq_len, self.config.position_encoding_maxlen))
        
        for pos in range(seq_len):
            for i in range(self.config.position_encoding_maxlen // 2):
                sine_pos, cosine_pos = self.pe(pos, i)
                
                self.positional_encoder[pos, 2 * i] = sine_pos
                self.positional_encoder[pos, 2 * i + 1] = cosine_pos
        # self.positional_encoder.reshape(seq_len, self.config.transformer_hidden_size)
        
        return self.positional_encoder

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config                

        self.shared_word_embedding = nn.Embedding(config.vocab_size, config.transformer_hidden_size)                
        self.encoder = TransformerEncoder(config, shared_word_embedding=self.shared_word_embedding)
        self.decoder = TransformerDecoder(config, shared_word_embedding=self.shared_word_embedding)

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        
        enc_output = self.encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
        dec_output = self.decoder(input_ids=dec_input_ids, enc_output=enc_output, enc_attention_mask=enc_attention_mask)
        
        return dec_output


class TransformerConfig:
    def __init__(self):
        self.vocab_size = 50265
        self.transformer_hidden_size = 512
        self.multi_head_num = 8
        self.position_encoding_maxlen = 512
        
        self.qkv_hidden_size = 64
                
        self.encoder_layer_num = 6
        self.decoder_layer_num = 6
        
class TransformerEncoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config                
                
        self.word_embedding = shared_word_embedding
        self.pos_embedding = PosEncoding(config)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.encoder_layer_num)])

    def forward(self, input_ids, attention_mask):
        
        input_repre = self.word_embedding(input_ids)
        input_repre += self.pos_embedding(input_repre.size(1))

        for layer in self.encoder_layers:
            input_repre = layer(input=input_repre, attention_mask=attention_mask)
            
        output = input_repre
        return output
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.multi_head_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.transformer_hidden_size)
        
        self.linear_1 = nn.Linear(config.transformer_hidden_size, config.transformer_hidden_size * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.transformer_hidden_size * 4, config.transformer_hidden_size)
        
    def forward(self, input, attention_mask):
        mha_output = self.layernorm(input + self.multi_head_attention(input=input, attention_mask=attention_mask))
        layer_output = self.layernorm(mha_output + self.linear_2(self.relu(self.linear_1(mha_output))))
        
        return layer_output    
        
class TransformerDecoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config                
                
        self.word_embedding = shared_word_embedding
        self.pos_embedding = PosEncoding(config)
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(config) for i in range(config.encoder_layer_num)])

    def forward(self, input_ids, enc_output, enc_attention_mask):
        
        input_repre = self.word_embedding(input_ids)
        input_repre += self.pos_embedding(input_repre.size(1))
        

        for layer in self.decoder_layers:
            input_repre = layer(input=input_repre, enc_output=enc_output, enc_attention_mask=enc_attention_mask)
            
        output = input_repre    
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.masked_attention = MultiHeadAttention(config)
        self.enc_dec_cross_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.transformer_hidden_size)
        
        self.linear_1 = nn.Linear(config.transformer_hidden_size, config.transformer_hidden_size * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.transformer_hidden_size * 4, config.transformer_hidden_size)
        
    def forward(self, input, enc_output, enc_attention_mask):
        
        masked_mha_output = self.layernorm(input + self.masked_attention(input=input, 
                                                                             attention_mask=None, 
                                                                             encoder_output=None))
        
        cross_mha_output = self.layernorm(masked_mha_output + self.enc_dec_cross_attention(input=masked_mha_output,
                                                                                        attention_mask=enc_attention_mask,
                                                                                        encoder_output=enc_output))
        layer_output = self.layernorm(cross_mha_output + self.linear_2(self.relu(self.linear_1(cross_mha_output))))
        
        return layer_output        


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, qkv, mask):
        query, key, value = qkv
        
        # spliting head
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # attention score + masking
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        
        if mask is None:
            # mask for masked self-attention
            batch_size, seq_len = query.size(0), query.size(2)
            mask = torch.ones(batch_size, seq_len).unsqueeze(-1).expand(-1, -1, seq_len).unsqueeze(1)
            mask = mask.triu(1)            
        else:
            mask = mask.unsqueeze(-1).expand(-1, -1, mask.size(1)).unsqueeze(1)
            
        masked_attention_score = attention_score.masked_fill(mask, -1e9)
        
        # attention weight
        attention_weight = nn.functional.softmax(masked_attention_score, -1)
        
        # splited attention output
        splited_attention_output = torch.matmul(attention_weight, value)
        
        # combined attention output
        attention_output = self.combine_heads(splited_attention_output)
        
        return attention_output
        
    def split_heads(self, tensor):
        batch_size, seq_len = tensor.size(0), tensor.size(1)
        tensor = tensor.unsqueeze(-1).reshape(batch_size, seq_len, self.config.multi_head_num, -1).transpose(1, 2)
        # tensor: [batch_size, num_heads, q or k seq_len, qkv_hidden_size]
        
        return tensor
        
    def combine_heads(self, tensor):
        batch_size, seq_len = tensor.size(0), tensor.size(2)
        tensor = tensor.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.query_layer = nn.Linear(self.config.transformer_hidden_size, self.config.qkv_hidden_size * self.config.multi_head_num, bias=False)
        self.key_layer = nn.Linear(self.config.transformer_hidden_size, self.config.qkv_hidden_size * self.config.multi_head_num, bias=False)
        self.value_layer = nn.Linear(self.config.transformer_hidden_size, self.config.qkv_hidden_size * self.config.multi_head_num, bias=False)
        self.output_layer = nn.Linear(self.config.transformer_hidden_size, self.config.transformer_hidden_size, bias=False)
        self.attention_layer = MultiHeadAttentionLayer(self.config)
        
        self.init_weight(self.query_layer)
        self.init_weight(self.key_layer)
        self.init_weight(self.value_layer)
        
    def init_weight(self, param):
        nn.init.xavier_uniform_(param.weight.data)

    def forward(self, input, attention_mask=None, encoder_output=None):
        # query, key, value matrix
        query = self.query_layer(input)
        if encoder_output is None:
            # self-attention
            key = self.key_layer(input)
            value = self.value_layer(input)
        else:
            # cross-attention
            key = self.key_layer(encoder_output)
            value = self.value_layer(encoder_output)
        
        attention_output = self.attention_layer((query, key, value), attention_mask)
        output = self.output_layer(attention_output)

        return output
            
model_config = TransformerConfig()
model = Transformer(config=model_config)

enc_input_ids_rand = torch.randint(0, 10, (5, 30))
enc_attention_mask = torch.randint(0, 2, (5, 30))

dec_input_ids_rand = torch.randint(0, 10, (5, 30))

output = model(enc_input_ids=enc_input_ids_rand,  
               enc_attention_mask=enc_attention_mask,
               dec_input_ids=dec_input_ids_rand)

print(output.shape)
