"""
File that contains code for the data management
"""
import os
import numpy as np
import openpyxl as opxl


def count_records(session, column_defaults, csv_file, field_delimiter):
    """
    Function that is used to count elements of a given csv file that will later be used.
    :param session: (Tensor) Tensorflow session
    :param column_defaults: List[Tensors] types of inputs that we get from the csv file
    :param csv_file: (str) path to csv file
    :return:
    """
    # Todo To Be Coded
    record_count = 0

    # example_dataset = tf.data.experimental.CsvDataset(
    #     filenames=[csv_file], record_defaults=column_defaults, header=True, field_delim=field_delimiter)
    #
    # example_iterator = tf.compat.v1.data.make_initializable_iterator(example_dataset)
    #
    # example_element = example_iterator.get_next()
    # example_initializer = example_iterator.initializer
    #
    # session.run(example_initializer)
    #
    # record_count = 0
    #
    # while True:
    #     try:
    #         _ = session.run(example_element)
    #         record_count += 1
    #     except tf.errors.OutOfRangeError:
    #         break

    return record_count


# todo evaluate if i really need this ?
def create_one_hot_vector(index, num_classes):
    """
    Function that creates a one hot encoding vector
    :param index: (int) index at which we would like to add the 1
    :param num_classes: (int) dimension of our vector
    :return:
    """
    label = np.zeros(num_classes)
    np.put(label, index, 1)

    return label


def dump_info(label_dictionary, data_dictionary, counter_dictionary, output_file, template_path):
    """
    Function that dumps metric information into an excel file for analysis
    :param label_dictionary: (dict)project label dictionary
    :param data_dictionary: (dict) data dictionary from evaluation
    :param counter_dictionary: (dict) counter of images per label
    :param output_file: (str) desired path for results
    :param template_path: (str) path of the template file
    :return: dump all information about a model's evaluation in an excel file
    """

    # Load template workbook
    wb = opxl.load_workbook(template_path, keep_vba=True)

    # First we create worksheets for a given label in a label dictionary
    for i in range(len(label_dictionary) - 1):
        buffer_sheet = wb["Classifieur Binaire"]
        wb.copy_worksheet(buffer_sheet)

    # Rename worksheets
    for en, sh in enumerate(wb):
        sh.title = str(label_dictionary[en])

    # sorting following the predicted class output
    for k in data_dictionary:
        data_dictionary[k].sort(key=lambda tup: tup[2])
        data_dictionary[k].reverse()
        sheet = wb[k]
        for index, info in enumerate(data_dictionary[k]):
            sheet["A" + str(index + 2)].value = info[0]
            sheet["B" + str(index + 2)].value = info[1]
            sheet["C" + str(index + 2)].value = info[2]

        sheet["D" + str(2)].value = "0.5"
        sheet["D" + str(51)].value = counter_dictionary[k]

    # Computation of Specificity which is equal to TN/(TN+FP)
    for k in data_dictionary:
        sheet = wb[k]
        negatives = "+".join(["%s!$D$51" % label for label in label_dictionary.values() if label != k])
        false_positives = "-".join(["$E$31", "$G$31"])
        # Number of True Negatives of class k
        true_negatives = "(" + negatives + "-" + "(" + false_positives + ")" + ")"
        sheet["F" + str(51)].value = "=100*" + true_negatives + "/" + "(" + negatives + ")"

    wb.save(output_file)
    wb.close()
