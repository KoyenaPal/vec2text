import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# 1. Load the JSON file
with open("profiles_docs.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. Process each item
processed_data = [
    {
        "sentence": " ".join(item["docs"]),
        "names": item["name"],
        "is_name": 1
    }
    for item in raw_data
]

# 3. Create the dataset
dataset = Dataset.from_list(processed_data)

# Optional: If you'd like to have a train/test split
# dataset = DatasetDict({"train": dataset})

# 4. Push to Hugging Face Hub (as private)
dataset_name = "koyena/profile_names"
dataset.push_to_hub(dataset_name, private=True)
