import json
import random
from huggingface_hub import upload_file
from pathlib import Path

# Data splitting configuration
input_file = Path(__file__).parent / 'data' / 'foamgpt_all.jsonl'
train_file = Path(__file__).parent / 'data' / 'foamgpt_train.jsonl'
test_file = Path(__file__).parent / 'data' / 'foamgpt_test.jsonl'
test_ratio = 0.1

# Hugging Face configuration
repo_id = "LeoYML/FoamGPT"

def split_data():
    """Split data into training and test sets"""
    print("Starting data splitting...")
    
    # Set random seed for reproducibility
    random.seed(0)
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - test_ratio))
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]

    # Write files
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"Data splitting completed: Train {len(train_data)} samples, Test {len(test_data)} samples")
    return train_file, test_file

def upload_to_huggingface(train_file, test_file):
    """Upload files to Hugging Face"""
    print("Starting upload to Hugging Face...")
    
    upload_file(
        path_or_fileobj=train_file,
        path_in_repo=train_file.name,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Uploaded training file: {train_file}")

    upload_file(
        path_or_fileobj=test_file,
        path_in_repo=test_file.name,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Uploaded test file: {test_file}")
    
    print("All files uploaded successfully!")

if __name__ == "__main__":
    # Execute data splitting
    print("Splitting data...")
    train_file, test_file = split_data()
    
    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    upload_to_huggingface(train_file, test_file)

