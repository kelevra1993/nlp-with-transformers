"""
The following script will contain the definition of a text classifier based on transformer
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch import nn
from math import sqrt


def scaled_dot_product_attention(query, key, value, mask):
    """
    Function that performs self attention given a query, key and value matrix.
    :param query: query matrix (input hidden state projected into generally smaller query space)
    :param key: key matrix (input hidden state projected into generally smaller query space)
    :param value: value matrix (input hidden state projected into generally smaller query space)
    :param mask: mask matrix (Generally a triangular matrix)
    :return:
    """
    dimension_key = key.size(-1)

    scores = torch.bmm(key, query.transpose(1, 2)) / sqrt(dimension_key)

    if mask is not None:
        # Now remove scores of tokens that come after a given token, for every token
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    return torch.bmm(weights, value)


# Creation of the initial token embeddor and positional embeddor.
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.embedding_dimension)
        self.positional_embeddings = nn.Embedding(config.max_position_embedding, config.embedding_dimension)
        self.layer_normalization = nn.LayerNorm(config.embedding_dimension, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        """
        Function that fetches the input embedding as well as the positional embedding based on the input_ids.
        :param input_ids: input ids of the tokenized data.
        :return:
        """
        # Create position IDs for input sequence
        sequence_length = input_ids.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)

        # Create Token and Position Embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_normalization(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# Implementation of the Attention Head
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_matrix = nn.Linear(config.embedding_dimension, config.head_dimension)
        self.key_matrix = nn.Linear(config.embedding_dimension, config.head_dimension)
        self.value_matrix = nn.Linear(config.embedding_dimension, config.head_dimension)

        # Kept just for analysis but not necessary
        self.weights_sizes = (config.embedding_dimension, config.head_dimension)

    def forward(self, hidden_state):
        """
        Function that computes self attention based on the provided hidden state.
        Note that we feed the projected hidden state in the query, key and value space.
        :param hidden_state: input matrix, either from the token and position embedding of from a transfomer block
        :return:
        """
        attention_outputs = scaled_dot_product_attention(query=self.query_matrix(hidden_state),
                                                         key=self.key_matrix(hidden_state),
                                                         value=self.value_matrix(hidden_state),
                                                         mask=None)
        return attention_outputs


# Create the MultiHead Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_attention_heads = config.embedding_dimension // config.head_dimension
        self.attention_heads = nn.ModuleList([AttentionHead(config) for _ in range(self.number_attention_heads)])
        self.linear_output_matrix = nn.Linear(self.number_attention_heads * config.head_dimension,
                                              config.embedding_dimension)

    def forward(self, hidden_state):
        """
        Function that applies multi head attention give the hidden state,
        either from the token and position embedding of from a transfomer block
        :param hidden_state: input matrix, either from the token and position embedding of from a transfomer block
        :return:
        """
        output = torch.cat([attention_head(hidden_state) for attention_head in self.attention_heads],
                           axis=-1)
        return self.linear_output_matrix(output)


# Now we create the Feed Forward layer, that takes in the concatenated ouput of the multihead attention
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded_dimension_size = int(config.reasoning_factor * config.embedding_dimension)
        self.first_linear_matrix = nn.Linear(config.embedding_dimension, expanded_dimension_size)
        self.second_linear_matrix = nn.Linear(expanded_dimension_size, config.embedding_dimension)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_probability)

    def forward(self, hidden_state):
        """
        Function that applies feed forward this is done after the Multi Head Attention Layer.
        :param hidden_state: input matrix from the Multi Head Attention Layer
        :return:
        """
        print(f"First Linear Feed Forward Size : {self.first_linear_matrix.weight.size()}")
        hidden_state = self.first_linear_matrix(hidden_state)
        # Introduce non-linearity between matrix multiplications
        hidden_state = self.gelu(hidden_state)

        print(f"Second Linear Feed Forward Size : {self.second_linear_matrix.weight.size()}")
        hidden_state = self.second_linear_matrix(hidden_state)
        hidden_state = self.dropout(hidden_state)

        return hidden_state


# Create the Transformer encoder layer by adding the normalization layers
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first_normalization_layer = nn.LayerNorm(config.embedding_dimension)
        self.multi_attention_head = MultiHeadAttention(config)
        self.second_normalization_layer = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward_layer = FeedForward(config)

    def forward(self, hidden_state):
        """
        Function that runs the encoder layer on the hidden state input,
        either from the token and position embedding of from a transfomer block.
        Goes through two normalization steps as well as two residuals inserting steps.
        :param hidden_state:
        :return:
        """
        # Normalize then apply muli-attention head
        intermidiate_output = self.first_normalization_layer(hidden_state)
        intermidiate_output = self.multi_attention_head(intermidiate_output)

        # Add residual
        intermidiate_output = intermidiate_output + hidden_state

        # Normalize then apply feed forward layer
        output = self.second_normalization_layer(intermidiate_output)
        output = self.feed_forward_layer(output)

        # Add residual
        output = output + intermidiate_output

        return output
