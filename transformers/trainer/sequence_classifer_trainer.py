"""
File that contains code for the Trainer Class
"""
import os
import sys
import torch
import time
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
from os.path import basename, dirname
# We have not created our own tokenizer therefore we will import one for simplification purposes.
# The goal being to start out with a simple text classification task and then build up from there.
from transformers import AutoTokenizer

module_folder = dirname(dirname(__file__))
sys.path.append(module_folder)

# Here we get the sequence classifier model that was coded in the model.py file
from models.model import TransformerForSequenceClassification

from utils import (make_dir, print_green, safe_dump, print_yellow, print_red, print_dictionary, get_trackers,
                   create_progress_bar, print_blue, print_bold, plot_and_save_confusion_matrix, compute_accuracy,
                   console_log_update_tracker)

from data_manager.data_utils import dump_info


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
         self.param_file, self.template_path, self.tensorboard_path,
         self.result_evaluation_file) = self.initialize_project_paths()

        # Get the tokenizer
        # todo later specify the path to the tokenizer ?
        self.tokenizer = self.get_tokenizer(self)

        # Get the device
        self.device = self.get_device()

        # Get The Classification Model
        self.model = TransformerForSequenceClassification(config=self.project_configuration)

        # Add the model to the device
        self.model.to(self.device)

        # Set up the loss
        self.loss = self.get_loss(self)

        # Setup the optimizer
        self.optimizer = self.get_optimizer(self)

        # Inform the user of the model size
        self.print_model_size()

        # Get Tokenized Iterators
        (self.training_dataset, self.validation_dataset, self.test_dataset,
         self.training_iterator, self.validation_iterator, self.test_iterator,
         self.test_iterations, self.label_dictionary) = self.get_train_validation_test_sets(self)

        # get trackers
        self.tracker_dictionary = get_trackers()

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
        model_path = os.path.join(self.project_configuration.PROJECT_FOLDER, raw_parameters)
        result_path = os.path.join(model_path, "Results")
        weight_path = os.path.join(model_path, "Weights")
        param_file = os.path.join(model_path, "params.json")
        template_path = os.path.join(module_folder, "data_manager", "Template.xlsx")
        tensorboard_path = os.path.join(self.project_configuration.PROJECT_FOLDER, "Tensorboard")

        # Results of validation on saved Models
        result_evaluation_file = os.path.join(model_path, "Results.txt")

        return raw_parameters, model_path, result_path, weight_path, param_file, template_path, tensorboard_path, result_evaluation_file

    def prepare_project_folder(self):
        """
        Function that create project folders in oder to launch training
        :return:
        """

        make_dir(self.weight_path)
        make_dir(self.result_path)
        make_dir(self.tensorboard_path)

        train_logger_directory = os.path.join(self.tensorboard_path, "Training")
        validation_logger_directory = os.path.join(self.tensorboard_path, "Validation")

        training_writer = SummaryWriter(train_logger_directory)
        validation_writer = SummaryWriter(validation_logger_directory)

        # Saving Training parameters to a "params.json" file if it does not exist
        if not (os.path.exists(self.param_file)):
            print_green("Creating params.json that contains Training and Validation Hyperparameters !!!\n")
            safe_dump(training_parameters=self.training_variables, destination=self.param_file)
        else:
            print_yellow("params.json already exists and does not need to be updated !!!\n")

        # Write the graph of the model
        training_writer.add_graph(self.model, input_to_model=torch.randn(self.project_configuration.batch_size,
                                                                         self.project_configuration.max_tokens).int())

        return training_writer, validation_writer

    @staticmethod
    def get_train_validation_test_sets(self):
        """
        Function that is used to get the emotion dataset. If the datasets has not already been stored locally we download it,
        and then we save in order to load it faster afterwards.
        :return:
        """

        training_dataframe = pd.read_csv(self.project_configuration.train_csv_file)
        validation_dataframe = pd.read_csv(self.project_configuration.valid_csv_file)
        test_dataframe = pd.read_csv(self.project_configuration.test_csv_file)

        training_dataframe["input_ids"] = training_dataframe["input_ids"].apply(ast.literal_eval)
        validation_dataframe["input_ids"] = validation_dataframe["input_ids"].apply(ast.literal_eval)
        test_dataframe["input_ids"] = test_dataframe["input_ids"].apply(ast.literal_eval)

        training_dataset, validation_dataset, test_dataset = (Dataset.from_pandas(training_dataframe),
                                                              Dataset.from_pandas(validation_dataframe),
                                                              Dataset.from_pandas(test_dataframe))

        training_dataset = DataLoader(training_dataset, batch_size=self.project_configuration.batch_size,
                                      shuffle=False,
                                      collate_fn=self.collate_function)
        validation_dataset = DataLoader(validation_dataset, batch_size=self.project_configuration.batch_size,
                                        shuffle=False,
                                        collate_fn=self.collate_function)
        test_dataset = DataLoader(test_dataset,
                                  shuffle=False,
                                  batch_size=1,
                                  collate_fn=self.collate_function)

        test_iterations = test_dataset.__len__()

        training_iterator = iter(training_dataset)
        validation_iterator = iter(validation_dataset)
        test_iterator = iter(test_dataset)

        # Set up the label dictionary
        label_dictionary = self.project_configuration.label_dictionary

        return training_dataset, validation_dataset, test_dataset, training_iterator, validation_iterator, test_iterator, test_iterations, label_dictionary

    @staticmethod
    def device_message_success(model_device):
        """
        Function that prints success message for the given device
        :param model_device: (str) device specified by the user
        :return:
        """
        print_green(f"{model_device} device detected and will be used for Training and Inference", add_separators=True)

    def get_device(self):
        """
        Function that is used to get the device that is set by the user
        :return:
        """

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
            device = None
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

        # Create project folders and setup tensorboard
        training_writer, validation_writer = self.prepare_project_folder()

        # Printing Model Summary
        self.model_and_iterations_summary()

        # Restore last model variables
        starting_iteration = self.restore_last_model() + 1

        progress_bar = None

        for iteration in range(starting_iteration, self.project_configuration.num_iterations + 1):

            try:
                # Re-Initialize Optimizer
                self.optimizer.zero_grad()

                self.training_iterator, training_loss, _, _, _, _ = self.run_model_and_update_trackers(
                    set_type="training",
                    iterator=self.training_iterator,
                    dataset=self.training_dataset,
                    summary_writer=training_writer,
                    iteration=iteration)

                # Run the backpropagation and optimizer
                training_loss.backward()
                self.optimizer.step()

                # Validation process done after optimizer step()
                if self.project_configuration.keep_validation_iterations:
                    with torch.no_grad():
                        self.validation_iterator, _, _, _, _, _ = self.run_model_and_update_trackers(
                            set_type="validation",
                            iterator=self.validation_iterator,
                            dataset=self.validation_dataset,
                            summary_writer=validation_writer,
                            iteration=iteration)

                # Dump information to console log at constant intervals
                if iteration % self.project_configuration.info_dump == 0:
                    progress_bar = self.manage_progress_bar(progress_bar)

                    self.tracker_dictionary = console_log_update_tracker(
                        iterations=iteration,
                        tracker_dictionary=self.tracker_dictionary,
                        info_dump=self.project_configuration.info_dump,
                        keep_validation_iterations=self.project_configuration.keep_validation_iterations)

                # Launching the evaluation on the test set and also saving the model
                if iteration % self.project_configuration.weight_saver == 0:
                    progress_bar = self.manage_progress_bar(progress_bar)

                    # Launching evaluation on the complete test data
                    self.launch_evaluation_on_single_model(
                        iteration=iteration,
                        inference_during_training=True)

                    self.save_model(iteration=iteration)

                # Initialise the progress bar
                if iteration == starting_iteration or iteration % self.project_configuration.info_dump == 0 or iteration % self.project_configuration.weight_saver == 0:
                    progress_bar = self.manage_progress_bar(progress_bar, iteration=iteration)
                else:
                    progress_bar.update(1)

            except KeyboardInterrupt:
                if progress_bar:
                    progress_bar.close()
                    time.sleep(1)

                self.save_model(iteration=iteration)

                exit()

    def manage_progress_bar(self, progress_bar, iteration=None):
        """
        Function that manages progress bar.
        :param progress_bar: (tqdm progress bar)
        :param iteration: (int) training iteration at which we are at.
        :return:
        """

        if progress_bar:
            progress_bar.update(1)
            progress_bar.close()
            progress_bar = None
            time.sleep(1)
        if iteration:
            # Stop the training process if we have reached the end.
            if iteration == self.project_configuration.num_iterations:
                exit()

            gap = min(self.project_configuration.info_dump - iteration % self.project_configuration.info_dump,
                      self.project_configuration.weight_saver - iteration % self.project_configuration.weight_saver)

            return create_progress_bar(total=gap, description=f"Training Iteration {iteration} to {iteration + gap}")

    def run_model_and_update_trackers(self, set_type, iterator, dataset, summary_writer=None, iteration=None):
        """
        Function that is used to run the model and update trackers for a defined iterator.
        :param set_type: (str) indicate whether we are dealing with training or validation.
        :param iterator: (iterator) iterator that we would like to go through.
        :param dataset: (dataset) dataset that is used.
        :param summary_writer: (SummaryWriter) summary writer for tensorboard
        :param iteration: (int) iteration
        :return:
        """
        iterator, sequence, input_ids, labels = self.get_iterator_batch(
            iterator=iterator,
            dataset=dataset,
            ignore_stop=False if set_type == "test" else True)

        # Run the inputs through the neural network
        logits, softmax = self.model(input_ids)

        # Needs to be verified thouroughly
        loss = self.loss(logits, labels)

        accuracy = compute_accuracy(predictions=logits.detach().cpu().numpy(), labels=labels)

        if summary_writer:
            summary_writer.add_scalar('loss', loss, iteration)
            summary_writer.add_scalar('accuracy', accuracy, iteration)

        # Update loss and accuracy trackers (only for training and validation cases)
        self.tracker_dictionary[f"{set_type}_moving_loss"] += loss
        self.tracker_dictionary[f"{set_type}_moving_accuracy"] += accuracy

        return iterator, loss, sequence, input_ids, labels, softmax

    def launch_evaluation_on_single_model(self, iteration, inference_during_training=False):
        """
        Function that runs inference on the complete test data and stores the results in an excel file
        while displaying the results to the terminal.
        :param iteration: (int) model iteration
        :param inference_during_training: (bool) boolean indicating if we are in the training phase or not.
        If so, write the results to a txt file containing quality metrics of previous models;
        :return:
        """

        start_test_timer = time.time()

        counter_dictionary, data_dictionary, confusion_matrix = self.initialize_counters(
            label_dictionary=self.project_configuration.label_dictionary,
            num_classes=self.project_configuration.num_labels)

        # Reset Test trackers
        for key in list(self.tracker_dictionary.keys()):
            if "test" in key:
                self.tracker_dictionary[key] = 0.0

        for test_iteration in tqdm(range(self.test_iterations), desc="Model Evaluation"):
            with torch.no_grad():
                # Keep going until we reach the end of the test dataset.
                self.test_iterator, _, test_sequence_text, test_input_ids, test_labels, test_softmax = self.run_model_and_update_trackers(
                    set_type="test",
                    iterator=self.test_iterator,
                    dataset=self.test_dataset)

                counter_dictionary, confusion_matrix, data_dictionary = self.update_test_counters(
                    counter_dictionary=counter_dictionary,
                    confusion_matrix=confusion_matrix,
                    data_dictionary=data_dictionary,
                    sequence_text=test_sequence_text[0],
                    sequence_label=test_labels.detach().cpu().numpy()[0],
                    softmax=test_softmax.detach().cpu().numpy()[0])

        # Re initialize the test iterator
        self.test_iterator = iter(self.test_dataset)

        # Get test accuracy
        test_accuracy = 100 * (self.tracker_dictionary["test_moving_accuracy"] / self.test_iterations)

        # Get list of messages that will be displayed to the user containing recalls per class
        # as well as averaged recall.
        message_list = self.get_accuracy_and_recall_messages(
            iteration=iteration,
            test_accuracy=test_accuracy,
            counter_dictionary=counter_dictionary)

        # Only write to result file during training
        if inference_during_training:
            target = open(self.result_evaluation_file, "a")
            for message in message_list:
                target.write(message + '\n')
            target.close()

        for message in message_list:
            if "Recall" in message:
                print_blue(message)
            elif "--" in message:
                print_yellow(message)
            else:
                print_bold(message)

        print(f"The Test Process Took {int((time.time() - start_test_timer))} seconds")

        # Create folder that will store results for a particular iteration
        iteration_result_folder = os.path.join(self.result_path, f"Iteration_{iteration}")
        make_dir(iteration_result_folder)

        # Then we dump information to a text file
        dump_info(self.label_dictionary,
                  data_dictionary=data_dictionary,
                  counter_dictionary=counter_dictionary,
                  template_path=self.template_path,
                  output_file=os.path.join(iteration_result_folder, f"Results_{iteration}.xlsx"))

    # TODO Make sure that we do indeed get to the end of our dataset and we do go through all of the inputs
    def get_iterator_batch(self, iterator, dataset, ignore_stop=True):
        """
        Function that is used to get the batch of interest, either for the Training, Validation and Test Set.
        :param iterator: (iterator) iterator that we would like to go through
        :param dataset: (dataset) dataset that is used to initialize the iterator. This is done when we reach the end of our dataset
        :param ignore_stop: (bool) boolean that is used to indicate whether or not to keep iterating on the dataset.
        This is useful when we want to loop over the dataset over and over for training and validation, but not necesarily
        for the test set where we would like to only have on pass in order to store the results for a given model iteration.
        :return:
        """
        try:
            batch = next(iterator)
        except StopIteration:

            iterator = iter(dataset)
            if ignore_stop:
                batch = next(iterator)
            else:
                return iterator, None, None, None

        # Get Input Ids and Labels
        # Do we need to transform it into a numpy array ?
        sequence = batch["text"]
        input_ids = torch.Tensor(batch["input_ids"]).to(self.device).int()
        labels = torch.Tensor(batch["merged_label"]).to(self.device).long()

        return iterator, sequence, input_ids, labels

    @staticmethod
    def initialize_counters(label_dictionary, num_classes):
        """
        # Setting dictionaries that keep track of classification prediction for metric computations and displays
        :param label_dictionary: dictionary containing the names of the desired classes
        :param num_classes: number of classes that are being predicted
        :return:
        """
        counter_dictionary = {}

        data_dictionary = {}
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for i in range(len(label_dictionary)):
            counter_dictionary[label_dictionary[i]] = {'total': 0, 'correct': 0}
            data_dictionary[label_dictionary[i]] = []

        return counter_dictionary, data_dictionary, confusion_matrix

    def update_test_counters(self, counter_dictionary, confusion_matrix,
                             data_dictionary, sequence_text, sequence_label, softmax):
        """
        Function that is used to update test counter dictionaries in order to later display and store them after model
        at a given iteration has been stored.
        :param counter_dictionary: (dict) dictionary containing counts of elements per class
        :param confusion_matrix: (dict) dictionary containing data that is used to create the confusion matrix
        :param data_dictionary: (dict) dictionary containing detailed predictions
        :param sequence_text: (str) sequence name string
        :param sequence_label: [float,float] labels
        :param softmax: [float,float] softmax applied on predictions
        :return:
        """

        # Fine grained evaluation
        # Dealing with a given label
        # Todo will definely have to be checked, since we have not yet checked that there is only one element.
        label_index = int(sequence_label)
        predicted_index = np.argmax(softmax, 0)
        counter_dictionary[self.label_dictionary[label_index]]["total"] += 1

        # Add element to confusion matrix
        confusion_matrix[predicted_index][label_index] += 1

        # A Correct prediction
        if predicted_index == label_index:
            data_dictionary[self.label_dictionary[label_index]].append(
                (sequence_text, "Right", softmax[predicted_index]))
            counter_dictionary[self.label_dictionary[label_index]]["correct"] += 1

        # A False prediction
        else:
            data_dictionary[self.label_dictionary[predicted_index]].append(
                (sequence_text, "False", softmax[predicted_index]))

        return counter_dictionary, confusion_matrix, data_dictionary

    @staticmethod
    def get_accuracy_and_recall_messages(iteration, test_accuracy, counter_dictionary):
        """
        Function that gets the message containing recall and accuracy information for the test set. The goal is to
        properly highlight it ot the user.
        :param iteration: (int) iteration of the model that we are currently looking at
        :param test_accuracy: (float) accuracy evaluated on the whole test dataset
        :param counter_dictionary:(dict) dictionary containing number of samples per P1 property
        :return:
        """
        message_list = [50 * "-", f"Model Iteration {iteration} Global Accuracy : {np.round(test_accuracy, 2)}%"]
        class_recalls = []

        for key, counter in counter_dictionary.items():
            class_recall = counter['correct'] / counter['total']
            message_list.append(f"Recall On {key.capitalize()} Class Is : {np.round(100 * class_recall, 2)}%")
            class_recalls.append(100 * class_recall)
        message_list.append(f"Average Recall On All Classes Is {np.round(np.mean(class_recalls), 2)}%")
        message_list.append(50 * "-")

        return message_list

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

    def print_model_size(self):
        """
        Prints the estimated size of the model and optionally stops training if specified.
        """
        # Calculate model parameter size
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        # Calculate model buffer size
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # Total model size in megabytes
        size_all_mb = (param_size + buffer_size) / 1024 ** 2

        # Print model size
        message = f"Estimated Model Size Without Optimizer : {size_all_mb:.2f} MB"
        print_yellow(message, add_separators=True)

        # Stop training message
        if self.project_configuration.compute_model_size:
            print_red("To Continue Training or Inference: Please Set compute_model_size To False",
                      add_separators=True)
            exit()

    def model_and_iterations_summary(self, training=True):
        """
        Function that print out triaining summary
        :param: training (boolean) variable that indicates that we are in the training phase
        :return: test_iterations (int) number of elements in our test dataset
        """

        # Fetch the number of smile sequences in our test dataset
        print(f"Model File : {self.raw_parameters}")

        if training:
            print(f"Number Of Training Iterations : {self.project_configuration.num_iterations}")
        print(f"Number Of Validation/Test Iterations : {self.test_iterations} \n")

    def extract_last_model_iteration(self):
        """
        Function that is used to get the last model that was saved for a given run. Useful when we are resuming
        training.
        :return:
        """
        last_iteration = 0

        full_checkpoint_logger = os.path.join(self.weight_path, "full-checkpoint")

        if not os.path.exists(full_checkpoint_logger):
            return last_iteration

        with open(full_checkpoint_logger, 'r') as file:
            first_line = file.readline().strip()

        last_iteration_string = first_line.split('"')[1]
        last_iteration = int(last_iteration_string.split("_")[-1])

        return last_iteration

    def restore_last_model(self, index_iteration=None):
        """
        Function that is used to restore the last model that was saved, if we are resuming training.
        If we are dealing with inference for validation or test set, to restore the model at index_iteration for evaluation.
        :param index_iteration: (int) iteration of interest for the model that we would like to restore
        :return:
        """

        if not index_iteration:
            index_iteration = self.extract_last_model_iteration()

        # Despite trying to get the last model None was found
        if not index_iteration:
            print_green(
                "No Initiation Model Weights Will Be Used...\nWe generate A New Model That Will Be Trained From Scratch.",
                add_separators=True)
        else:

            self.load_model(iteration=index_iteration)
            print_green(f"We Loaded A Model That Was Previously Saved At Iteration {index_iteration}",
                        add_separators=True)
        return index_iteration

    def load_model(self, iteration):
        """
        Function that is used to restore a model that was previously saved, either in order to continue
        training or to run inference.
        :param iteration: (int) iteration at which the model is being saved.
        :return:
        """

        model_path = os.path.join(self.weight_path, f"Iteration_{iteration}.pth")

        loaded_checkpoint = torch.load(model_path)

        self.model.load_state_dict(loaded_checkpoint["model_state"])

    def save_model(self, iteration):
        """
        # Function that is used to save the model
        :param iteration: (int) iteration at which the model is being saved.
        :return:
        """
        print_yellow(f"Model is being saved at iteration {iteration}...")
        checkpoint = {"model_state": self.model.state_dict()}
        model_path = os.path.join(self.weight_path, f"Iteration_{iteration}.pth")
        torch.save(checkpoint, model_path)
        self.dump_in_checkpoint(iteration)
        print_green("Model has been saved successfully")

    def dump_in_checkpoint(self, iteration):
        """
        Function that is used to dump information about the saved checkpoint in a file for later retrieval.
        :param iteration: (int) iteration at which the model is being saved.
        :return:
        """
        checkpoint_file = os.path.join(self.weight_path, "full-checkpoint")

        # Saved the iteration in a checkpoint file
        try:
            with open(checkpoint_file, "r") as f:
                d = f.readlines()
        except FileNotFoundError:
            d = []

        with open(checkpoint_file, "w") as f:
            if len(d) == 0:
                d.append(f'model_checkpoint_path: "Iteration_{iteration}"\n')
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            else:
                d[0] = f'model_checkpoint_path: "Iteration_{iteration}"\n'
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            for line in d:
                f.write(line)


from utils import load_configuration_variables, print_red, print_yellow

# Load configuration object
project_configuration_object = load_configuration_variables(
    application_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    experiment_name="sequence_classifier_project_config.example.yml")

# Load the trainer
trainer = SequenceClassifierTrainer(project_configuration=project_configuration_object)

# Run training
with torch.device(trainer.device):
    # Train the model
    trainer.train()
