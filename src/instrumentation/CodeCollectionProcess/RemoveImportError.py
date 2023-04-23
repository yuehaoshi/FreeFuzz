import os

error_file = "ImportErrorList.txt"
error_files = set()

# Extract the file names from the error file
with open(error_file, "r") as f:
    for line in f:
        file_name = line.split()[1]
        file_name = os.path.basename(file_name)
        # print(file_name)
        error_files.add(file_name)

test_dir = "/Users/ashi_mac/VSC/CS527/Paddle/test"  # Change this to the test directory

# Delete the files in the test directory
for root, dirs, files in os.walk(test_dir):
    for file in files:
        # print("File: ", file)
        if file in error_files:
            file_path = os.path.abspath(os.path.join(root, file))
            print(f"Deleting file: {file_path}")
            print("File to be deleted: ", file_path)
            os.remove(file_path)
