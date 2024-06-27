"""
    This is the implementation of chapter 1 of the book natural language processing with Transformers.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers.tokenization_utils_base
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from umap import UMAP
from torch.nn.functional import cross_entropy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import neural_network

from utilities.utils import print_green, print_bold, print_blue, print_red, print_yellow

# Set the width of the output
np.set_printoptions(linewidth=500)

# Set column width for dataframes
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

# Load the emotions dataset
emotions = load_dataset("emotion")

# Load all of the datasets
training_dataset, validation_dataset, test_dataset = emotions["train"], emotions["validation"], emotions["test"]

# Set the format of the datasets to pandas ?
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()


def label_int2str(row):
    """
    Function that transforms an integer into a string, transforming the inter label to it's string equivalent
    :param row: row of a dataframe
    :return:
    """
    return emotions["train"].features["label"].int2str(row)


# Apply the string label to all of the rows
df["label_name"] = df["label"].apply(label_int2str)
df.head()

# Let's start training our model
# First we are only going to get a model and the add a classification layer
# that is the only part that is going to be trained
model_ckpt = "distilbert-base-uncased"

# Get the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = """We are learning about tokenizers and how they are use to train large language models."""

encoded_text = tokenizer(text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
input_masks = encoded_text.attention_mask

emotions.reset_format()


# Function that will be used to tokenize input
def tokenize(batch: dict) -> transformers.tokenization_utils_base.BatchEncoding:
    """
    Function that takes in as input a batch that is a dictionary and encodes all of the "text" values.
    :param batch: dictionary of texts as a batch can be multiple texts
    :return: encoded texts for a given batch
    """

    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape : {inputs['input_ids'].size()}")

# Remap the inputs so that we can add them to a device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = model(**inputs).last_hidden_state.cpu().numpy()


def extract_hidden_states(batch: dict) -> dict:
    """
    Function that run a model on recieved tokenized data to get last hidden states
    :param batch: dictonary that is of the form {input_ids:xxxx, attention_mask:xxxx, label:xxxx, text:xxxx}
    :return:
    """

    # Place the model inputs on the device
    model_inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

    # Extract that last hidden states
    with torch.no_grad():
        last_hidden_state = model(**model_inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


# Change the format of the emotions into torch format for input_ids, attention_mask and label.
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Just make the emotions_encoded variables a little bit shorter.
# emotions_encoded["train"] = emotions_encoded["test"]

print_yellow("runing computation of hidden states")
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True, batch_size=1500)
print_green("Computation of hidden states has been completed")

print(emotions_hidden.column_names)

# Transform data into numpy arrays
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
X_test = np.array(emotions_hidden["test"]["hidden_state"])

Y_train = np.array(emotions_hidden["train"]["label"])
Y_valid = np.array(emotions_hidden["validation"]["label"])
Y_test = np.array(emotions_hidden["test"]["label"])

# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)

# Initialize and fit UMAP
mapper = UMAP(metric="cosine").fit(X_scaled)

# Create a DataFrame of 2D Embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = Y_train

# # Just create figures with matplotlib for visualisation purposes
# fig, axes = plt.subplots(2, 3, figsize=(7, 5))
# axes = axes.flatten()
#
# cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
#
labels = emotions["train"].features["label"].names


#
# for i, (label, cmap) in enumerate(zip(labels, cmaps)):
#     df_emb_sub = df_emb.query(f"label == {i}")
#     axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
#     axes[i].set_title(label)
#     axes[i].set_xticks([]), axes[i].set_yticks([])
#
# plt.tight_layout()
# plt.show()

# # Train a logistic regression model
# lr_classifier = LogisticRegression(max_iter=3000)
# lr_classifier.fit(X_train, Y_train)
# lr_classifier.score(X_valid, Y_valid)
# print(f"The accuracy of the logisitic regression classifier is {lr_classifier.score(X_valid, Y_valid)}")


# # Train an MLPClassifier : Parallel code just for exploration.
# mlp_classifier = neural_network.MLPClassifier(max_iter=3000, hidden_layer_sizes=(400, 200, 100), early_stopping=False)
# mlp_classifier.fit(X_train, Y_train)
# mlp_classifier.score(X_valid, Y_valid)
# print(f"The accuracy of the multi layer perceptron classifier is {mlp_classifier.score(X_valid, Y_valid)}")

def plot_confusion_matrix(y_preds, y_true, labels, model_name):
    """
    # TODO Add docstring
    :param y_preds:
    :param y_true:
    :param labels:
    :param model_name:
    :return:
    """

    cm = confusion_matrix(y_true, y_preds, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title(f"Normalized Confusion Matrix For {model_name}")
    plt.show()


# y_preds = lr_classifier.predict(X_valid)
# plot_confusion_matrix(y_preds, Y_valid, labels, model_name="Logistic Regressor")

# Now we move on to fine tuning the whole model rather than just using the last hidden units output as a feature vector
num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))


def compute_metrics(pred):
    """
    # TODO Add docstring
    :param pred:
    :return:
    """
    ground_truth = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(ground_truth, preds, average="weighted")
    acc = accuracy_score(ground_truth, preds)

    return {"accuracy": acc, "f1": f1}


batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size  # iterations to reach an epoch
model_name = f"{model_ckpt}-finetuded-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer
                  )

saved_model_folder = "/Users/Robert/PycharmProjects/nlp-with-transformers/finetuned_model"

if os.path.exists(os.path.join(saved_model_folder, "config.json")):
    print_green("We found a pretrained model....")
    trainer._load_from_checkpoint(saved_model_folder)
    print_green("We loaded the pretrained model")
else:
    print_blue("Relaunching the training phase")
    trainer.train()
    print_green("The training phase is over")


# preds_output = trainer.predict(emotions_encoded["validation"])
#
# # # Save the model for later
# # trainer.save_model(output_dir=saved_model_folder)
#
# # Print out the metrics for the current model
# print(preds_output.metrics)
# y_preds = np.argmax(preds_output.predictions, axis=1)
# plot_confusion_matrix(y_preds, Y_valid, labels, model_name="Fine-Tuned-DistilBert")

def forward_pass_with_label(batch):
    """
    Function that applies the forward pass to a given batch, this will allow us to compute the loss
    and the predicted label
    :param batch:
    :return:
    """
    # Place all input tensors on the same device as the model

    model_inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

    # Add the model to the device
    model.to(device)

    with torch.no_grad():
        output = model(**model_inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")

    # Place outputs on CPU for compatibility with other dataset columns
    return {
        "logits": output.logits.cpu().numpy(),
        "loss": loss.cpu().numpy(),
        "predicted_label": pred_label.cpu().numpy()}


# Convert our dataset back to pyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Compute the loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label,
                                                                    batch_size=16,
                                                                    batched=True)

# Create a dataframe with the texts, losses and predicted/true labels
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss", "logits"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = df_test["predicted_label"].apply(label_int2str)

# Get the largest losses to understand the large differences
message = "".join([50 * "-", "Largest Losses", 50 * "-"])
print_red(message)
largest_losses = df_test.sort_values("loss", ascending=False).head(10)
print(largest_losses)
print_red(len(message) * "-")

# Get the smallest losses ot understand the "easy cases"
message = "".join([50 * "-", "Lowest Losses", 50 * "-"])
print_green(message)
lowest_losses = df_test.sort_values("loss", ascending=True).head(10)
print(lowest_losses)
print_green(len(message) * "-")
