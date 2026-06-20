import os

# Dataset path
yes_path = "../dataset/yes"
no_path = "../dataset/no"

# Count images
yes_count = len(os.listdir(yes_path))
no_count = len(os.listdir(no_path))

print("=" * 40)
print("BrainSight-AI Dataset Report")
print("=" * 40)

print(f"Tumor Images (YES): {yes_count}")
print(f"Normal Images (NO): {no_count}")
print(f"Total Images: {yes_count + no_count}")

print("=" * 40)