from urllib.request import urlretrieve
import os
import os.path as op
import zipfile
from pathlib import Path


S3_BUCKET = "https://s3.amazonaws.com/rytrose-personal-website/"


def get_nottingham():
    nottingham_path = "../../data/raw/nottingham-midi"
    nottingham_zip_name = "nottingham-midi.zip"
    if op.exists(nottingham_path):
        print("Nottingham exists.")
        return
    else:
        print("Fetching Nottingham...")
        os.mkdir(nottingham_path)

        # Get the zip file
        urlretrieve(S3_BUCKET + nottingham_zip_name, op.join(nottingham_path, nottingham_zip_name))

        # Unzip it
        with zipfile.ZipFile(op.join(nottingham_path, nottingham_zip_name), 'r') as zip_ref:
            zip_ref.extractall(nottingham_path)

        # Delete zip file
        os.remove(op.join(nottingham_path, nottingham_zip_name))

        print("fetched Nottingham.")


def get_bebop():
    bebop_path = "../../data/processed/songs"
    bebop_zip_name = "bebop-incomplete.zip"
    if op.exists(bebop_path):
        print("Bebop exists.")
        return
    else:
        print("Fetching Bebop...")
        os.mkdir(bebop_path)

        # Get the zip file
        urlretrieve(S3_BUCKET + bebop_zip_name, op.join(bebop_path, bebop_zip_name))

        # Unzip it
        with zipfile.ZipFile(op.join(bebop_path, bebop_zip_name), 'r') as zip_ref:
            zip_ref.extractall(bebop_path)

        # Delete zip file
        os.remove(op.join(bebop_path, bebop_zip_name))

        print("fetched Bebop.")


if __name__ == "__main__":
    get_nottingham()
    get_bebop()
