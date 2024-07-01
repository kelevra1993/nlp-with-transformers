"""
The following script will contain a class that will be used to configure our transformer
"""


class TransformerTextConfiguration:
    """
    Configuration class that will be used to set the text classifier model parameters
    """

    embedding_dimension = 768  # For simplicity same embedding dimension along all layers (size of the hidden states)
    head_dimension = 64  # embedding dimension of a single head (intrinsicly sets the number of attention heads)
    reasoning_factor = 3  # upscaling from the embedding dimension in Feed Forward(after multi-head attention layer)
    hidden_dropout_probability = 0.1  # percentage of dropped values after Feed Forward
    vocabulary_size = 30522  # Size of the vocabulary of the tokenizer
    max_position_embedding = 512  # maximum possible length of a sequence that the model
    num_hidden_layers = 12  # Number of (MHA + FF Layers) Equivalent of number of ConvBlocks in Computer Vision
    num_labels = 2  # Number of Labels


configuration_dictionary = TransformerTextConfiguration
