import gzip
import json

# Function to read the JSON data
def read_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Read the data
reviews_data = read_json_gz(file_path)

# Example: Print the first review
print(reviews_data[0])
