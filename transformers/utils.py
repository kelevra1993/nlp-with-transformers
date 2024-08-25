import os
import json
import yaml
import argparse
import time
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class ProjetConfiguration:
    """
    Class for project configuration for easier access later on
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# todo make sure that everything is being used
def get_configuration_object(application_folder, config):
    """
    Function that gets configuration variables and put them in a configuration object that can be passed to Trainer class,
    , data preparation functions, project folder preparation function ...e.t.c.
    :param application_folder: (str) path to the application folder
    :param config: (dict) dictionary containing specified configurations that one would like to use
    :return:
    """
    project_configuration_variables = {
        "PROJECT_FOLDER": os.path.join(application_folder, "trained_sequence_classifier_models",
                                       config["folder_structure"]["project_folder"]),
        "train_csv_file": os.path.join(application_folder, config["folder_structure"]["data_files"]["train"]),
        "valid_csv_file": os.path.join(application_folder, config["folder_structure"]["data_files"]["validation"]),
        "test_csv_file": os.path.join(application_folder, config["folder_structure"]["data_files"]["test"]),
        "field_delimiter": config["data"]["field_delimiter"],
        "label_dictionary": config["data"]["label_dictionary"],
        "num_labels": len(config["data"]["label_dictionary"]),
        "max_tokens": config["model"]["max_tokens"],
        "positional_encoder": config["model"]["positional_encoder"]["type"],
        "positional_encoder_parameters": config["model"]["positional_encoder"]["parameters"],
        "vocabulary_size": config["model"]["vocabulary_size"],
        "num_hidden_layers": config["model"]["transformer"]["num_hidden_layers"],
        "embedding_dimension": config["model"]["transformer"]["embedding_dimension"],
        "head_dimension": config["model"]["transformer"]["head_dimension"],
        "reasoning_factor": config["model"]["transformer"]["reasoning_factor"],
        "hidden_dropout_probability": config["model"]["transformer"]["hidden_dropout_probability"],
        "use_aggregator": config["model"]["use_aggregator"]["activated"],
        "aggregator_type": config["model"]["use_aggregator"]["type"],
        "aggregator_parameters": config["model"]["use_aggregator"]["parameters"],
        "fully_connected_sizes": config["model"]["fully_connected_sizes"],  # TODO To determine if will be used
        "scaler": config["model"]["scaler"],
        "device": config["training"]["device"],
        "batch_size": config["training"]["settings"]["batch_size"],
        "info_dump": config["training"]["settings"]["info_dump"],
        "keep_validation_iterations": config["training"]["settings"]["keep_validation_iterations"],
        "weight_saver": config["training"]["settings"]["weight_saver"],
        "num_iterations": config["training"]["settings"]["num_iterations"],
        "learning_rate": float(config["training"]["hyperparameters"]["learning_rate"]),
        "compute_model_size": config["training"]["alternative_modes"]["compute_model_size"],
        "view_input_sequences": config["training"]["alternative_modes"]["view_input_sequences"],
        "index_iteration": config["inference"]["index_iteration"],
        "error_dump": config["inference"]["error_dump"]
    }

    return ProjetConfiguration(**project_configuration_variables)


def load_configuration_variables(application_folder, experiment_name="sequence_classifier_project_config.example.yml"):
    """
    Function that return project configuration variables based on the experiment file that is being used
    :param application_folder: (str) path to the application folder
    :param experiment_name: (str) experiment file name
    :return: project_configuration_variables : (dict) variable containing project configuration variables
    """

    # Get Current Folder as well as configuration file in the setup folder
    library_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_file = os.path.join(library_folder, "config", experiment_name)

    # Load project configuration variables
    try:
        project_configuration_object = get_configuration_object(application_folder=application_folder,
                                                                config=yaml.safe_load(open(setup_file, 'r')))
    except KeyError:
        raise
    except:
        print_red(f"This Configuration File {setup_file} Does Not Exist ")
        exit()

    return project_configuration_object


def safe_dump(training_parameters, destination):
    """
    Function that dumps training parameters for traceability
    :param training_parameters: (dict) Dictionary containing training parameters
    :param destination: (str) Destination of the json file that will contain all the training parameters
    :return: Storage of params.json at destination path
    """
    try:
        with open(destination, "w") as fp:
            json.dump(training_parameters, fp, sort_keys=True, indent=4, separators=(",", ":"))
    except KeyboardInterrupt:
        safe_dump(training_parameters, destination)


def make_dir(path):
    """
    Function that creates a folder when there isn't one
    :param path: (str) Folder path that we want to create if non-existent
    :return: Folder creation if non-existent
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_blue(output, add_separators=False):
    """
    Prints the output string in blue color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[94m" + "\033[1m" + output + "\033[0m")
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output, add_separators=False):
    """
    Prints the output string in green color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[32m" + "\033[1m" + output + "\033[0m")
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output, add_separators=False):
    """
    Prints the output string in yellow color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[93m" + "\033[1m" + output + "\033[0m")
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output, add_separators=False):
    """
    Prints the output string in red color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[91m" + "\033[1m" + output + "\033[0m")
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output, add_separators=False):
    """
    Prints the output string in bold font.
    :param output: The string that we wish to print in bold font.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[1m" + str(length * "-") + "\033[0m")
        print("\033[1m" + output + "\033[0m")
        print("\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[1m" + output + "\033[0m")


def print_dictionary(dictionary, indent):
    json.dumps(dictionary, indent=indent)


# todo update documentation
def console_log_update_tracker(iterations, tracker_dictionary, info_dump, keep_validation_iterations):
    """
    :param iterations: (int) Global step during training for a given model
    :param tracker_dictionary: (dict) dictionary containing tracker information
    :param info_dump: (int) number of iteration that were ran from previous information dump
    """
    # todo use left and right for the print
    print("-------------------------------------------------------------")
    print(f"We called the model {iterations} times")
    print(
        f"Moving Average of Training Loss is       : {np.round((tracker_dictionary['training_moving_loss'].item() / info_dump), 2)}")
    if keep_validation_iterations:
        print(
            f"Moving Average of Validation Loss is     : {np.round((tracker_dictionary['validation_moving_loss'].item() / info_dump), 2)}")
    print(
        f"Moving Average of Training Accuracy is   : {np.round(100 * (tracker_dictionary['training_moving_accuracy'] / info_dump), 2)}%")
    if keep_validation_iterations:
        print(
            f"Moving Average of Validation Accuracy is : {np.round(100 * (tracker_dictionary['validation_moving_accuracy'] / info_dump), 2)}%")
    print("These %d Iterations took %d Seconds" % (info_dump, (time.time() - tracker_dictionary['start'])))
    print("-------------------------------------------------------------")

    return get_trackers()


def get_trackers():
    """
    Function that gets elements that we would like to tack during training process
    :return:
    """

    tracker_dictionary = {
        "start": time.time(),
        "training_moving_accuracy": 0.0,
        "validation_moving_accuracy": 0.0,
        "test_moving_accuracy": 0.0,
        "training_moving_loss": 0.0,
        "validation_moving_loss": 0.0,
        "test_moving_loss": 0.0}

    return tracker_dictionary


from tqdm import tqdm


def create_progress_bar(total, description):
    """
    Creates and returns a tqdm progress bar.
    :param total: (int) length of the progress bar
    :param description: (str) description of the progress bar
    :return:
    """
    time.sleep(1)
    return tqdm(total=total, desc=description, leave=True)


# todo document function
def compute_accuracy(predictions, labels):
    # Initialize a counter for correct predictions
    correct_predictions = 0

    # Iterate over the list of tuples (predicted probabilities) and the list of true labels
    for prediction, label in zip(predictions, labels):

        # Determine the predicted label (index with the highest probability)
        predicted_label = np.argmax(prediction)

        # Compare the predicted label with the true label
        if predicted_label == label:
            correct_predictions += 1

    # Calculate accuracy as the ratio of correct predictions to the total number of predictions
    accuracy = correct_predictions / len(labels)

    return accuracy


# Confusion Matrix Functions Still in progress
def get_new_figure(figure_title, figure_size=None):
    """
     Initialize graphics
    :param figure_title: (str) title of the figure
    :param figure_size: (tuple) size of the figure
    :return:
    """
    fig1 = plt.figure(figure_title, figure_size)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configure_cell_text_and_colors(array_dataframe, line, column, CellText, facecolors, posi, font_size, total_values):
    """
    Configure cell text and colors for a heatmap cell.
    :param array_dataframe: (DataFrame) The input DataFrame.
    :param line: (int) The row index of the cell.
    :param column: (int) The column index of the cell.
    :param CellText: (str) The text associated with the cell.
    :param facecolors: The facecolors of the cells.
    :param posi: The position index.
    :param font_size: (int) The font size.
    :param total_values: (int) The total number of values.
    :return: (list, list) A tuple containing two lists:
             - A list of text elements to add.
             - A list of text elements to delete.
    """
    text_to_add = []
    text_to_delete = []

    cell_value = array_dataframe[line][column]
    per = (float(cell_value) / total_values) * 100
    curr_column = array_dataframe[:, column]
    ccl = len(curr_column)

    # last line  and/or last column
    if (column == (ccl - 1)) or (line == (ccl - 1)):
        # totals and percents
        if cell_value != 0:
            if (column == ccl - 1) and (line == ccl - 1):
                tot_rig = 0
                for i in range(array_dataframe.shape[0] - 1):
                    tot_rig += array_dataframe[i][i]
                per_ok = (float(tot_rig) / cell_value) * 100
            elif column == ccl - 1:
                tot_rig = array_dataframe[line][line]
                per_ok = (float(tot_rig) / cell_value) * 100
            elif line == ccl - 1:
                tot_rig = array_dataframe[column][column]
                per_ok = (float(tot_rig) / cell_value) * 100
        else:
            per_ok = 0

        per_ok_s = ["%.2f%%" % (per_ok), "100%"][per_ok == 100]

        # text to delete
        text_to_delete.append(CellText)

        # text to add
        font_prop = matplotlib.font_manager.FontProperties(weight="bold", size=font_size)

        text_kwargs = dict(
            color=[0.7059, 0.7451, 0.9412, 1.0], ha="center", va="center", gid="sum", fontproperties=font_prop)

        lis_txt = ["%d " % cell_value, per_ok_s]
        lis_kwa = [text_kwargs]

        dic = text_kwargs.copy()
        dic["color"] = "lawngreen"
        lis_kwa.append(dic)

        lis_pos = [(CellText._x, CellText._y - 0.11), (CellText._x, CellText._y + 0.11)]

        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            text_to_add.append(newText)

        # set background color for sum cells (last line and last column)
        carr = [0.0, 0.098, 0.21, 1.0]
        if (column == ccl - 1) and (line == ccl - 1):
            carr = [0.0, 0.098, 0.19, 1.0]
        facecolors[posi] = carr

    else:
        txt = "%s \n  %.2f%%" % (cell_value, per)

        CellText.set_text(txt)
        CellText.set_color([0.0, 0.098, 0.25, 1.0])
        CellText.set_weight("bold")

        # main diagonal
        if column == line:
            # set background color in the diagonal to blue
            CellText.set_color([0.0, 0.098, 0.19, 1.0])
            facecolors[posi] = [0.4863, 0.988, 0.0, 1.0]

    return text_to_add, text_to_delete


def insert_totals(data_frame):
    """
    Functions that inserts totals on columns and lines
    :param data_frame: (DataFrame)
    :return:
    """

    sum_line = []
    for item_line in data_frame.iterrows():
        sum_line.append(item_line[1].sum())

    sum_column = []
    for column in data_frame.columns:
        sum_column.append(data_frame[column].sum())

    data_frame["Total \nPredictions \n Precision% "] = sum_line
    sum_column.append(np.sum(sum_line))
    data_frame.loc["Total Labels \nSensitivity%"] = sum_column


def pretty_plot_confusion_matrix(data_frame, figure_size=None, model_iteration=None, iteration_result_folder=None):
    """
    Function that creates a fancy confusion matrix
    :param data_frame: (DataFrame)
    :param figure_size: List(int) size of the figure that is the confusion matrix
    :param model_iteration: (int) iteration of the model that interests us
    :param iteration_result_folder: (str) path to folder, where to store the information
    :return:
    """

    x_label = "Labels"
    y_label = "Predictions"

    # Create "Total" Columns
    insert_totals(data_frame)

    # Default Confusion Matrix : Always print on same matrix
    fig, ax1 = get_new_figure("Default Confusion Matrix", figure_size)

    # thanks for seaborn
    ax = sn.heatmap(data_frame, annot=True, annot_kws={"size": 11}, linewidths=2, ax=ax1,
                    cbar=False, cmap="tab10_r", linecolor="w", fmt=".2f", )

    # Set Y Tick labels Orientation
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight="bold")

    # Bold Font For X labels
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")

    # Face Color List
    quadmesh = ax.findobj(matplotlib.collections.QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # Iteration on text elements
    array_dataframe = np.array(data_frame.to_records(index=False).tolist())
    total_values = array_dataframe[-1][-1]
    text_to_add = []
    text_to_delete = []
    position = -1  # from left to right, bottom to top.

    for text in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(text.get_position()) - [0.5, 0.5]
        line = int(pos[1])
        column = int(pos[0])
        position += 1

        # set text
        txt_results = configure_cell_text_and_colors(array_dataframe, line, column, text, facecolors, position, 11,
                                                     total_values=total_values, )
        text_to_add.extend(txt_results[0])
        text_to_delete.extend(txt_results[1])

    # Remove The Old Text
    for item in text_to_delete:
        item.remove()

    # Append New Text
    for item in text_to_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # Titles And Legends
    ax.set_title("Iteration %d Confusion Matrix" % model_iteration, fontweight="bold")
    ax.set_xlabel(x_label, fontweight="bold")

    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")

    ax.set_ylabel(y_label, fontweight="bold")
    plt.tight_layout()

    # Save Figure in the Result Folder
    plt.savefig(os.path.join(iteration_result_folder, "Confusion_Matrix_%d.png" % model_iteration))


def plot_and_save_confusion_matrix(confusion_matrix, model_iteration, iteration_result_folder, num_classes,
                                   label_dictionary):
    """
    # Confusion matrix imports
    # plot a pretty confusion matrix with seaborn
    # @author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
    # Link :https://github.com/wcipriano/pretty-print-confusion-matrix

    Function that plots and saves the confusion matrix of a given iteration
    :param label_dictionary:
    :param confusion_matrix: np.array -> our confusion matrix
    :param model_iteration: index / iteration of the model
    :param iteration_result_folder: path to model's result folder
    :param label_dictionary: (dict) dictionary containing the labels of interest
    :return: save a png confusion matrix
    """
    # Get Pandas Data Frame
    data_frame_confusion_matrix = pd.DataFrame(confusion_matrix, index=range(0, num_classes),
                                               columns=range(0, num_classes))

    data_frame_confusion_matrix = data_frame_confusion_matrix.rename(index=label_dictionary)
    data_frame_confusion_matrix = data_frame_confusion_matrix.rename(columns=label_dictionary)

    pretty_plot_confusion_matrix(
        data_frame=data_frame_confusion_matrix,
        figure_size=[num_classes + 5, num_classes + 5],
        model_iteration=model_iteration,
        iteration_result_folder=iteration_result_folder)
