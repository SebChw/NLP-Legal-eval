import zipfile
from io import BytesIO

import requests

from constants import DATA_PATH, TEST_URL, TRAIN_URL

if __name__ == "__main__":
    DATA_PATH.mkdir(exist_ok=True)
    for zip_file_url in [TRAIN_URL, TEST_URL]:
        r = requests.get(zip_file_url, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(DATA_PATH)
