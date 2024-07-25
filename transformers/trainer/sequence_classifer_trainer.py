"""
File that contains code for the Trainer Class
"""
import os
import torch
from tqdm import tqdm
import torch.optim as optimizer
from torch.utils.data import DataLoader

# Code that is used to load emotions datasets but will be completed deleted once we have downloaded it.
from datasets import load_dataset

# We have not created our own tokenizer therefore we will import one for simplification purposes.
# The goal being to start out with a simple text classification task and then build up from there.
from transformers import AutoTokenizer

# Here we get the sequence classifier model that was coded in the model.py file
from models.model import TransformerForSequenceClassification

from utils import (make_dir, print_green, safe_dump, print_yellow, print_red,
                   print_blue, print_bold, plot_and_save_confusion_matrix)


class SequenceClassifierTrainer:
    """
    Trainer class that can be used for training, evaluation and prediction
    """

    def __init__(self, **kwargs):

        # Get the project configuration
        self.project_configuration = kwargs.get("project_configuration")

        # Turn all variables to dictionary for storage
        self.training_variables = self.project_configuration.__dict__

        # Always initialise project paths containing model weights and results
        (self.raw_parameters, self.model_path, self.result_path, self.weight_path,
         self.param_file, self.template_path, self.tensorboard_path) = self.initialize_project_paths()

        # Get the tokenizer
        self.tokenizer = self.get_tokenizer(self)

        # Get the device
        self.device = self.get_device(self)

        # Get The Classification Model
        self.model = TransformerForSequenceClassification(config=self.project_configuration)

        # Add the model to the device
        self.model.to(self.device)

        # Set up the loss
        self.loss = self.get_loss(self)

        # Setup the optimizer
        self.optimizer = self.get_optimizer(self)

        # Get Tokenized Datasets
        (self.training_dataset,
         self.validation_dataset,
         self.test_dataset,
         self.test_iterations,
         self.label_dictionary) = self.get_train_validation_test_sets(self)

    def initialize_project_paths(self):
        """
        Function that initializes project paths that indicate where models weights are results are stored
        :return: raw_parameters (str), model_path (str), result_path (str), weight_path (str),
         param_file (str), template_path (str), tensorboard_path (str)
        """
        # Raw parameters for model information storage
        params = ""

        # First get the maximum tokens
        params += str(self.project_configuration.max_tokens) + "-"

        # Dealing with the positional encoder first
        params += self.project_configuration.positional_encoder.upper() + "-"

        if self.project_configuration.positional_encoder == "blstm":
            blstm_parameters = self.project_configuration.positional_encoder_parameters["blstm"]
            params += str(blstm_parameters["hidden_units"]) + "-"
            params += str(blstm_parameters["layers"]) + "-"

        # Dealing with the vocabulary size
        params += "VOC-" + str(self.project_configuration.vocabulary_size) + "-"

        # Dealing with the transformer blocks
        params += "TRANS-BLOCKS-" + str(self.project_configuration.num_hidden_layers) + "-"  # hidden layers
        params += "EMB-" + str(self.project_configuration.embedding_dimension) + "-"
        params += "SINGLE-HEAD-" + str(self.project_configuration.head_dimension) + "-"
        params += "REAS-" + str(self.project_configuration.reasoning_factor) + "-"

        if self.project_configuration.use_aggregator:
            agg_parameters = self.project_configuration.aggregator_parameters
            if self.project_configuration.aggregator_type == "blstm":
                params += "AGG-BLSTM-" + str(agg_parameters["blstm"]["hidden_units"]) + "-" + str(
                    agg_parameters["blstm"][
                        "layers"]) + "-"
            if self.project_configuration.aggregator_type == "squeeze-and-excite":
                params += "AGG-SE-" + str(agg_parameters["squeeze-and-excite"]["number_of_matrices"]) + "-"

        raw_parameters = params + "FC-" + '_'.join(map(str, self.project_configuration.fully_connected_sizes))

        # Model path for Result Analysis, Error Analysis, Model Storing, Model Freezing if non-existant
        model_path = os.path.join(self.project_configuration.PROJECT_FOLDER, "Models", raw_parameters)
        result_path = os.path.join(model_path, "Results")
        weight_path = os.path.join(model_path, "Weights")
        param_file = os.path.join(model_path, "params.json")
        template_path = os.path.join("servier", "data_manager", "Template.xlsx")
        tensorboard_path = os.path.join(self.project_configuration.PROJECT_FOLDER, "Tensorboard")

        return raw_parameters, model_path, result_path, weight_path, param_file, template_path, tensorboard_path

    def prepare_project_folder(self):
        """
        Function that create project folders in oder to launch training
        :return: result_evaluation_file (str) path to file containing training results
        """

        make_dir(self.weight_path)
        make_dir(self.result_path)
        make_dir(self.tensorboard_path)

        # Saving Training parameters to a "params.json" file if it does not exist
        if not (os.path.exists(self.param_file)):
            print_green("Creating params.json that contains Training and Validation Hyperparameters !!!\n")
            safe_dump(training_parameters=self.training_variables, destination=self.param_file)
        else:
            print_yellow("params.json already exists and does not need to be updated !!!\n")

        # Results of validation on saved Models
        result_evaluation_file = os.path.join(self.model_path, "Results.txt")

        return result_evaluation_file

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
                                          batch_size=self.project_configuration.batch_size)

        # Load all of the datasets
        training_dataset, validation_dataset, test_dataset = (tokenized_emotions["train"],
                                                              tokenized_emotions["validation"],
                                                              tokenized_emotions["test"])
        # Todo the shuffling part might need re-adjustment
        training_dataset = DataLoader(training_dataset, batch_size=self.project_configuration.batch_size,
                                      shuffle=False,
                                      collate_fn=self.collate_function)
        validation_dataset = DataLoader(validation_dataset, batch_size=self.project_configuration.batch_size,
                                        shuffle=False,
                                        collate_fn=self.collate_function)

        test_dataset = DataLoader(test_dataset,
                                  shuffle=False,
                                  collate_fn=self.collate_function)
        test_iterations = test_dataset.__len__()

        # Set up the label dicitonary
        label_dictionary = {str(index): label for index, label in
                            enumerate(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])}

        return training_dataset, validation_dataset, test_dataset, test_iterations, label_dictionary

    @staticmethod
    def device_message_success(device):
        """
        Function that prints success message for the given device
        :param device: (str) device specified by the user
        :return:
        """
        message = f"{device} device has been detected and will be used for Training and Inference"
        print_green(len(message) * '-')
        print_green(message)
        print_green(len(message) * '-')

    @staticmethod
    def get_device(self):
        """
        Function that is used to get the device that is set by the user
        :param self:
        :return:
        """
        device = None

        # TODO Test out with cpu to see if it is really slow....
        # TODO Find out how to manage the choice of the gpu, when there are multiple gpus in a machine.

        if self.project_configuration.device == "cpu":
            device = torch.device("cpu")
            self.device_message_success("cpu")
        elif self.project_configuration.device == "gpu":
            # Get CUDA Compatibility if we are dealing with Nvidia Graphics Card
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.device_message_success("cuda")
            else:
                print_red("You chose a CUDA GPU, but it is not available or visible, Try installing CUDA")
                print_yellow(f"Or Choose A Device Among 'cpu'(slow) or 'mps'(for macos)")
                exit()
        elif self.project_configuration.device == "mps":
            # Get Metal Device Compatibility For Mac M1 and above
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.device_message_success("mps")
        else:
            print_red(f"Could Not Find Device -> {self.project_configuration.device}")
            print_yellow(f"Choose A Device Among 'cpu', 'gpu' or 'mps'")

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
        """
        Training function that runs a for loop for self.project_configuration.index_iterations amount of index_iterations
        :return:
        """

        # Create project folders
        result_evaluation_file = self.prepare_project_folder()

        # TODO MUST DO : SETUP TENSORBOARD !!!
        # Code for tensorboard

        # Printing Model Summary
        self.model_and_iterations_summary()

        # Computation of the running loss
        running_loss = 0.0

        while True:
            # Todo to be coded to integrate starting from a saved model.
            index = 0
            if index > self.project_configuration.num_iterations:
                break
            # Go through training iterations
            for batch in self.training_dataset:

                # testing out the dataloader to see the limitations
                if index > self.project_configuration.num_iterations:
                    break

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

                # update index
                index += 1
                # Todo Will be changed afterwards, to be moved to another function
                if (index + 1) % self.project_configuration.info_dump == 0:
                    print(100 * '-')
                    print(f"We Are At Iteration {index + 1} Of Training")
                    print(
                        f"The Average Loss Over The Last {self.project_configuration.info_dump} Iterations Is {running_loss / self.project_configuration.info_dump:.4f}")
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
        """
        Function that gets our optimizer we will be using the Adam Optimiser
        :param self:
        :return:
        """
        return optimizer.Adam(self.model.parameters(), lr=self.project_configuration.learning_rate)

    def model_and_iterations_summary(self, training=True):
        """
        Function that print out triaining summary
        :param: training (boolean) variable that indicates that we are in the training phase
        :return: test_iterations (int) number of elements in our test dataset
        """

        # Fetch the number of smile sequences in our test dataset
        print(f"\nModel File : {self.raw_parameters}\n")

        if training:
            print(f"Number Of Training Iterations : {self.project_configuration.num_iterations}")
        print(f"Number Of Validation/Test Iterations : {self.test_iterations} \n")


# TODO Will later be removed
from utils import load_configuration_variables, print_red, print_yellow

# Load configuration object
project_configuration_object = load_configuration_variables(
    application_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    experiment_name="sequence_classifier_project_config.example.yml")

# Load the trainer
trainer = SequenceClassifierTrainer(project_configuration=project_configuration_object)

# Run training
device = torch.device("mps")
with torch.device(device):
    # Train the model
    trainer.train()
