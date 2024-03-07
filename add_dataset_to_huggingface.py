from datasets import DatasetDict, load_dataset

data = load_dataset("json", data_files="Data/discord_stella.jsonl")

dataset_name = "dandelion4/ex-1"
data.push_to_hub(dataset_name, branch="main", private=True)
