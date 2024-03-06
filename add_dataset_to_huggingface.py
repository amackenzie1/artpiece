from datasets import DatasetDict, load_dataset

data = load_dataset("json", data_files="Data/discord.jsonl")

print(data)

dataset_name = "dandelion4/ex-3"
data.push_to_hub(dataset_name, branch="main", private=True)
