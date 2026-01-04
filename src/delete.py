import shutil
import nltk
import os

# Find where NLTK is storing the broken files
for path in nltk.data.path:
    if os.path.exists(path):
        print(f"Deleting corrupted folder: {path}")
        try:
            shutil.rmtree(path)
            print("Successfully deleted!")
        except Exception as e:
            print(f"Error deleting: {e}")

print("---")
print("Now RESTART YOUR KERNEL and run your import again.")