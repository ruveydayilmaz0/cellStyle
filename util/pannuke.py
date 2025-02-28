from datasets import load_dataset

ds = load_dataset("RationAI/PanNuke")

# Extract an image from the dataset
image = ds["fold1"][0]["image"]
a = 1
