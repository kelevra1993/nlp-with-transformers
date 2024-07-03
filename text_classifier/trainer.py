"""
The purpose of the following script is to train a model to be able to classify text from twitter,
either a positive text (joyful) or negative text( sad, angry, jealous, e.t.c.)
"""
import torch
from datasets import load_dataset

# We have not created our own tokenizer therefore we will import one for simplification purposes.
# The goal being to start out with a simple text classification task and then build up from there.
from transformers import AutoTokenizer


class Trainer:
    """
    Trainer class that can be used for training, evaluation and prediction
    """

    def __init__(self, **kwargs):
        # Get Datasets
        (self.training_dataset,
         self.validation_dataset,
         self.test_dataset,
         self.label_dictionary) = self.get_train_validation_test_sets()

        return None

    @staticmethod
    def get_train_validation_test_sets(self):
        """
        Function that is used to get the emotion dataset. It is already properly formatted so we can just get
        the training set, validation set and test set from the dictionary
        :return:
        """
        # Load the emotions dataset
        emotions = load_dataset("emotion")

        # Load all of the datasets
        training_dataset, validation_dataset, test_dataset = emotions["train"], emotions["validation"], emotions["test"]

        # Set up the label dicitonary
        label_dictionary = {str(index): label for index, label in
                            enumerate(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])}

        return training_dataset, validation_dataset, test_dataset, label_dictionary

    # TODO To be documented
    # @staticmethod
    # def get_tokenizer(self):
    #     """
    #     Function that is used to get the tokenizer.
    #     :param self:
    #     :return:
    #     """
    #     # Get the model tokenizer: Initially only using
    #     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #
    #     return tokenizer
    #
    # # Function that will be used to tokenize input
    # def tokenize(batch: dict) -> transformers.tokenization_utils_base.BatchEncoding:
    #     """
    #     Function that takes in as input a batch that is a dictionary and encodes all of the "text" values.
    #     :param batch: dictionary of texts as a batch can be multiple texts
    #     :return: encoded texts for a given batch
    #     """
    #
    #     return tokenizer(batch["text"], padding=True, truncation=True)
    #
    #
    # def train(self):
    #     # Insertion of for loop and get metrics
    #     return None
