import requests
import os


dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
dataset_filename = 'wdbc.data'
data_folder = 'data'


os.makedirs(data_folder, exist_ok=True)

response = requests.get(dataset_url)
if response.status_code == 200:
    with open(os.path.join(data_folder, dataset_filename), 'wb') as f:
        f.write(response.content)
    print(f"Dataset downloaded successfully as {dataset_filename}")
else:
    print(f"Failed to download dataset: Status code {response.status_code}")
