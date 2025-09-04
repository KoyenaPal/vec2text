#!/usr/bin/env python3

import torch
import os
import sys

# Add the vec2text directory to the path
sys.path.append('/share/u/koyena/rep-inverse/vec2text')

from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

# Based on the configuration from run.py
# These are the default values that would be used for the checkpoint

# Model arguments
model_args = ModelArguments(
    model_type="t5-base",  # This seems to be the model used based on the checkpoint path
    embedder_model_name="EleutherAI/pythia-14m",
    use_frozen_embeddings_as_input=False,
    embeddings_from_layer_n=None,  # Not using hidden layers
    embedder_no_grad=False,  # Using gradients
)

# Data arguments  
data_args = DataArguments(
    dataset_name="person_finder",
    use_less_data=-1,  # Not using tiny data
)

# Save the files
checkpoint_dir = "/share/u/koyena/rep-inverse/vec2text/copy_of_layer_3_till_end_run/replaced_names_data_fine_tuned_version__EleutherAI__pythia-14m__person_finder__from_gradients/checkpoint-239800"

print(f"Saving model_args.bin to {checkpoint_dir}")
torch.save(model_args, os.path.join(checkpoint_dir, "model_args.bin"))

print(f"Saving data_args.bin to {checkpoint_dir}")
torch.save(data_args, os.path.join(checkpoint_dir, "data_args.bin"))

print("Files created successfully!")
print(f"model_args: {model_args}")
print(f"data_args: {data_args}") 