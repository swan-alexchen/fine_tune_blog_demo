from datasets import Dataset
import json

def get_qa_dataset():
    with open('data/nebula_block_qa.json', 'r') as f:
        raw_data = json.load(f)

    dataset_data = []
    
    # Extract conversations from each item in raw_data
    for item in raw_data:
        if "conversations" in item and item["conversations"][0]["from"] == "human" and item["conversations"][1]["from"] == "gpt":
            dataset_data.append({
                "conversations": item["conversations"]
            })

    return Dataset.from_list(dataset_data)


if __name__ == "__main__":
    dataset = get_qa_dataset()
    print("Generated Dataset:")
    print(dataset)
    # You can save the dataset if needed, for example:
    # dataset.save_to_disk("fiction_qa_dataset")
