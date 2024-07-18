"""
The purpose of the following script is to train a model to be able to classify text from twitter,
either a positive text (joyful) or negative text( sad, angry, jealous, e.t.c.)
"""
import torch
from torchsummary import summary
import torch.optim as optimizer

from datasets import load_dataset

# We have not created our own tokenizer therefore we will import one for simplification purposes.
# The goal being to start out with a simple text classification task and then build up from there.
from transformers import AutoTokenizer

# Here we get the classifier model that was coded in the model.py file
from model import TransformerForSequenceClassification

# Here we get the configuration of the model
from config import configuration_dictionary

from tqdm import tqdm
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class that can be used for training, evaluation and prediction
    """

    def __init__(self, **kwargs):

        # Get the configuration dictionary
        self.configuration_dictionary = kwargs.get("configuration_dictionary")

        # Get the tokenizer
        self.tokenizer = self.get_tokenizer(self)

        # Get Tokenized Datasets
        (self.training_dataset,
         self.validation_dataset,
         self.test_dataset,
         self.label_dictionary) = self.get_train_validation_test_sets(self)

        # Get the device
        self.device = self.get_device(self)

        # Get The Classification Model
        self.model = TransformerForSequenceClassification(config=configuration_dictionary)

        # Add the model to the device
        self.model.to(self.device)

        # Set up the loss
        self.loss = self.get_loss(self)

        # Setup the optimizer
        self.optimizer = self.get_optimizer(self)

    @staticmethod
    def get_train_validation_test_sets(self):
        """
        Function that is used to get the emotion dataset. It is already properly formatted so we can just get
        the training set, validation set and test set from the dictionary
        :return:
        """
        # Load the emotions dataset
        emotions = load_dataset("emotion")

        # Tokenize the dataset.
        tokenized_emotions = emotions.map(self.tokenize_and_merge_classes, batched=True,
                                          batch_size=self.configuration_dictionary.batch_size)

        # Load all of the datasets
        training_dataset, validation_dataset, test_dataset = (tokenized_emotions["train"],
                                                              tokenized_emotions["validation"],
                                                              tokenized_emotions["test"])
        # Todo the shuffling part might need re-adjustment
        training_dataset = DataLoader(training_dataset, batch_size=self.configuration_dictionary.batch_size,
                                      shuffle=False,
                                      collate_fn=self.collate_function)
        validation_dataset = DataLoader(validation_dataset, batch_size=self.configuration_dictionary.batch_size,
                                        shuffle=False,
                                        collate_fn=self.collate_function)
        test_dataset = DataLoader(test_dataset, batch_size=self.configuration_dictionary.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collate_function)
        # Set up the label dicitonary
        label_dictionary = {str(index): label for index, label in
                            enumerate(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])}

        return training_dataset, validation_dataset, test_dataset, label_dictionary

    @staticmethod
    def get_device(self):
        # TODO Test out with cpu to see if it is really slow....
        device = torch.device("cpu")

        # Get CUDA Compatibility if we are dealing with Nvidia Graphics Card
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # Get Metal Device Compatibility For Mac M1 and above
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device("mps")

        return device

    @staticmethod
    def get_tokenizer(self):
        """
        Function that is used to get the tokenizer. We will be using the distil-bert tokenizer for simplicity purpose.
        :param self:
        :return:
        """
        # Get the model tokenizer: Initially only using
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        return tokenizer

    # Transform the labels, to only have two labels
    @staticmethod
    def transform_label(self, label):
        """
        Sadness, Anger and Fear are all merged into a negative set 0
        Joy, Love and Surprise are alle merged into a positive set 1
        :param self:
        :param label:
        :return:
        """
        if label in [0, 3, 4]:
            return 0

        if label in [1, 2, 5]:
            return 1

    # Function that will be used to tokenize input and merge classes
    def tokenize_and_merge_classes(self, batch: dict):
        """
        Function that takes in as input a batch that is a dictionary and encodes all of the "text" values.
        :param batch: dictionary of texts as a batch can be multiple texts
        :return: encoded texts for a given batch
        """
        output = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors='pt')
        output.update({"merged_label": [self.transform_label(self, label=label) for label in batch["label"]]})

        return output

    @staticmethod
    def collate_function(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}

    def train(self):

        # Insertion of for loop and get metrics
        for epoch in range(self.configuration_dictionary.number_epochs):

            # Computation of the running loss
            running_loss = 0.0

            # Go through training iterations
            for index, batch in enumerate(self.training_dataset):

                # Get Input Ids and Labels
                training_input_ids = torch.Tensor(batch["input_ids"])
                training_labels = torch.Tensor(batch["label"])
                training_merged_labels = torch.Tensor(batch["merged_label"])

                # Do we need to transform it into a numpy array ?
                # Add them to the device
                training_input_ids = training_input_ids.to(self.device)
                training_input_ids = training_input_ids.int()
                training_labels = training_labels.to(self.device)  # Not necessarily needed for this line
                training_merged_labels = training_merged_labels.to(self.device)

                # Re-Initialize Optimizer
                self.optimizer.zero_grad()

                # Run the inputs through the neural network
                outputs = self.model(training_input_ids)

                # Needs to be verified thouroughly
                loss = self.loss(outputs, training_merged_labels)

                # Run the backpropagation and optimizer
                loss.backward()
                self.optimizer.step()

                # TODO Retest to see if the model does backpropagation correctly
                #  normally the loss should be lower and depending on the learning rate
                #  we should properly predict the output class for previous input elements

                # WRITE CODE HERE

                # Add to the running loss
                running_loss += loss.item()

                # Todo Will be changed afterwards, to be moved to another function
                info_dump = 200
                if (index + 1) % info_dump == 0:
                    print(100 * '-')
                    print(f"We Are At Iteration {index + 1} Of Epoch {epoch}")
                    print(f"The Average Loss Over The Last {info_dump} Iterations Is {running_loss / info_dump:.4f}")
                    print(100 * '-')
                    running_loss = 0

    @staticmethod
    def get_loss(self):
        """
        Function that is used in order to get the cross entropy loss.
        Note that input to the cross entropy loss are the raw prediction that have not had softmax applied on them.
        The function will apply the softmax and compute the loss.
        :param self:
        :return:
        """
        return torch.nn.CrossEntropyLoss()

    @staticmethod
    def get_optimizer(self):
        return optimizer.Adam(self.model.parameters(), lr=0.0001)


trainer = Trainer(configuration_dictionary=configuration_dictionary)
device = torch.device("mps")
with torch.device(device):
    # Train the model
    trainer.train()
