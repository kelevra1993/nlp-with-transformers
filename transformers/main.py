import os
import torch
from trainer.sequence_classifer_trainer import SequenceClassifierTrainer
from utils import load_configuration_variables

# Load configuration object
project_configuration_object = load_configuration_variables(
    application_folder=os.path.dirname(os.path.abspath(__file__)),
    experiment_name="sequence_classifier_project_config.example.yml")

# Load the trainer
trainer = SequenceClassifierTrainer(project_configuration=project_configuration_object)

# Run training
with torch.device(trainer.device):
    # Train the model
    trainer.train()
