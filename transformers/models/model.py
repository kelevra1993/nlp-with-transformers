"""
The following script will contain the definition of a text classifier based on transformer
"""
import math
import torch
import torch.nn.functional as F
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


class PositionalEncoding(nn.Module):
    def __init__(self, device, embedding_dimension, max_length=512, n=10000):
        """
        Function that initializes our positional encoder matrix, based on users configuration inputs.
        Based on the sine and cosine representation of position.
        :param embedding_dimension: (int) dimension of the token embedding
        :param max_length: (int) maximum length of the sequence
        :param n: (int) n
        """
        # inherit from Module
        super().__init__()

        # create tensor of 0s
        positional_encoder = torch.zeros(max_length, embedding_dimension)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # compute divisor for positional encoding
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2) * -(math.log(n) / embedding_dimension))

        # compute sine on even indices
        positional_encoder[:, 0::2] = torch.sin(k * div_term)

        # compute cosine on odd indices
        positional_encoder[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        self.positional_encoder = positional_encoder.unsqueeze(0)
        self.positional_encoder = self.positional_encoder.to(torch.device(device))

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("sine_cosine_positional_encoder_matrix", self.positional_encoder)

    def forward(self, x):
        """

        :param x: embeddings (batch_size, seq_length, embedding_dimension)
        :return: embeddings + positional encodings (batch_size, seq_length, embedding_dimension)
        """

        # add positional encoding to the embeddings
        x += self.positional_encoder[:, : x.size(1)].requires_grad_(False)

        return x


class BinaryPositionalEncoding(nn.Module):
    def __init__(self, device, embedding_dimension, max_length=512):
        """
        Function that initializes our positional encoder matrix, based on users configuration inputs.
        Based on binary encoding representation
        :param embedding_dimension: (int) dimension of the token embedding
        :param max_length: (int) maximum length of the sequence
        """
        super().__init__()

        # Create a tensor to hold the positional encodings
        positional_encoder = torch.zeros(max_length, embedding_dimension)

        for pos in range(1, max_length + 1):
            # Convert the position to binary
            binary_repr = [int(b) for b in bin(pos)[2:]]

            # Add leading zeros to match the embedding dimension size
            binary_repr = [0] * (embedding_dimension - len(binary_repr)) + binary_repr

            # Fill the positional encoder with the binary representation
            positional_encoder[pos - 1] = torch.tensor(binary_repr, dtype=torch.float32)

        # Add an extra dimension and move to the specified device
        self.positional_encoder = positional_encoder.unsqueeze(0).to(device)

        # Register the positional encoding as a buffer, so it’s not a model parameter
        self.register_buffer("binary_positional_encoder_matrix", self.positional_encoder)

    def forward(self, x):
        """
        :param x: embeddings (batch_size, seq_length, embedding_dimension)
        :return: embeddings + positional encodings (batch_size, seq_length, embedding_dimension)
        """

        # Add the positional encoding to the input embeddings
        x += self.positional_encoder[:, :x.size(1)].requires_grad_(False)

        return x


# Creation of the initial token embeddor and positional embeddor.
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.embedding_dimension)

        if config.positional_encoder == "sine-cosine":
            self.positional_embeddings = PositionalEncoding(
                config.device,
                config.embedding_dimension,
                max_length=config.max_tokens,
                n=config.parameters)
        elif config.positional_encoder == "binary":
            self.positional_embeddings = BinaryPositionalEncoding(
                config.device,
                config.embedding_dimension,
                max_length=config.max_tokens
            )
        else:
            self.positional_embeddings = None

        self.layer_normalization = nn.LayerNorm(config.embedding_dimension, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        """
        Function that fetches the input embedding as well as the positional embedding based on the input_ids.
        :param input_ids: input ids of the tokenized data.
        :return:
        """

        # Create Token Embedding
        embeddings = self.token_embeddings(input_ids)

        if self.positional_embeddings:
            # Combine token and position embeddings
            embeddings = self.positional_embeddings(embeddings)

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
        # Todo mask will eventually need to be tested
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
        output = torch.cat([attention_head(hidden_state) for attention_head in self.attention_heads], axis=-1)
        return self.linear_output_matrix(output)


# Now we create the Feed Forward layer, that takes in the concatenated ouput of the multihead attention
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded_dimension_size = int(config.reasoning_factor * config.embedding_dimension)
        self.first_linear_matrix = nn.Linear(config.embedding_dimension, expanded_dimension_size)
        self.second_linear_matrix = nn.Linear(expanded_dimension_size, config.embedding_dimension)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_probability)

    def forward(self, hidden_state):
        """
        Function that applies feed forward this is done after the Multi Head Attention Layer.
        :param hidden_state: input matrix from the Multi Head Attention Layer
        :return:
        """
        hidden_state = self.first_linear_matrix(hidden_state)
        # Introduce non-linearity between matrix multiplications
        hidden_state = self.activation(hidden_state)

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


# Put everything together to create the ModelEncoder
class ModelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.architecture = config.architecture.upper()
        self.embeddings = Embeddings(config)
        if self.architecture == "TRANSFORMER":
            self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        if self.architecture == "LSTM":
            self.layers = nn.ModuleList([LSTMEncoder(config)])

        self.output_to_consider = self.get_output_to_consider(self)

    @staticmethod
    def get_output_to_consider(self):
        # TODO Evaluate taking the same output for both the Transformer and the LSTM.
        #  For the Transformer normally it shouldn't change anything if we take the first or the last element
        #  It seems arbitrary for the Transformers, but it might also be the case for the LSTM.
        """
        Function that sets the output to consider, if we are dealing with transformers we take the first element,
        if we are dealing with LSTMs, we take the last element.
        :param self:
        :return:
        """
        if self.architecture == "TRANSFORMER":
            return 0
        if self.architecture == "LSTM":
            return -1

    def forward(self, x):
        """
        Function that takes the input ids as tokens, runs the initial embedding issued from the token embedding
        and the positional embedding then passes them through the transformer blocks.
        :param x: input token ids from a tokenizer
        :return:
        """
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x)

        # Keep the hidden state of the first token for the Transformer
        # Keep the hidden state of the last output for the LSTM
        return x[:, self.output_to_consider, :]


# Create A Classifier, based either on Transformers or Bi-LSTM encoder
class ModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_encoder = ModelEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_probability)

        self.fully_connected_block = None
        input_size = config.embedding_dimension

        # Add Fully Connected Layers if defined
        if config.fully_connected_sizes:
            self.fully_connected_layers = []

            for size in config.fully_connected_sizes:
                self.fully_connected_layers.append(nn.Linear(input_size, size))
                self.fully_connected_layers.append(nn.ReLU())
                input_size = size

            self.fully_connected_block = nn.Sequential(*self.fully_connected_layers)

        self.classifier = nn.Linear(input_size, config.num_labels)

    def forward(self, x):
        """
        Function that passes inputs to the encoder model
        applies dropout on it and then classifies it.
        :param x: input token ids from a tokenizer
        :return:
        """
        x = self.model_encoder(x)
        x = self.dropout(x)

        if self.fully_connected_block:
            x = self.fully_connected_block(x)

        x = self.classifier(x)

        return x, F.softmax(x, dim=-1)


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.device = config.device
        self.input_size = config.embedding_dimension
        self.bidirectional = config.bidirectional
        self.hidden_size = config.embedding_dimension // 2 if self.bidirectional else config.embedding_dimension
        self.num_layers = config.num_lstm_layers
        self.number_state_layers = self.num_layers * 2 if self.bidirectional else self.num_layers

        # Configuration of the LSTM
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional)

    def forward(self, x):
        """
        Forward function for the Long Short Term Memory Layers
        Input is of size batch_size,sequence_length,embedding_dimension
        Output will be of size batch_size,sequence_length,embedding_dimension
        Output dimension has been standardised, in order to manage the potential
        bidirectionality of the LSTM. The hidden size is divided by 2 when bidirectonality is activated
        :param x: input from the embedding + positional encoding layer
        :return:
        """

        hidden_state_0 = torch.zeros(self.number_state_layers, x.size(0), self.hidden_size).to(self.device)
        cell_state_0 = torch.zeros(self.number_state_layers, x.size(0), self.hidden_size).to(self.device)

        output, _ = self.lstm(x, (hidden_state_0, cell_state_0))

        return output
