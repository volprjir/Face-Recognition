# Author Jiri Volprecht

import click
import os
import cv2
import shutil
import random
import time
from faces_train import train_faces


def unique_id():
    """
    Generator for the unique ids
    :return: Returns unique ids
    """
    seed = random.getrandbits(32)
    while True:
        yield seed
        seed += 1


def create_dataset(dataset_dir, count, camera):
    """
    Generates dataset from the connected camera
    :param dataset_dir: Directory to store the output.
    :param count: Number of frames to capture.
    :param camera: Camera ID for opencv lib.
    :raise Exception: If camera does not work.
    """
    if input("Are you ready to take pictures? y/n: ") == "n":
        exit(0)
    video = cv2.VideoCapture(camera)
    cnt = 0
    print(f"Turning on the camera with id {camera} to take {count} frames. Smile :)...")
    unique_seq = unique_id()
    while cnt != count:
        filename = f"{dataset_dir}{next(unique_seq)}.png"

        check, frame = video.read()
        if not check:
            raise Exception("Camera does not work!")
        print(f"{cnt}/{count}: Capturing: {filename}....")
        cv2.imwrite(filename, frame)
        time.sleep(2)
        cnt += 1
    print("Dataset created successfully!")
    video.release()


def check_basedir(dir_to_check):
    """
    Check the passed directory.
    :param dir_to_check: Folder to check
    :raise Exception: If dataset base folder does not exist
    """
    print(f"Checking directory: {dir_to_check} ...")
    if not os.path.isdir(dir_to_check):
        raise Exception("Dataset folder does not exists!")


def process_dataset_directory(base_dir, name, clean):
    """
    Prepare the dataset folder. Clean it or create it if necessary.
    :param base_dir: Base directory for storing output
    :param name: Name of dataset
    :param clean: Should remove all files in it
    :return: Final dataset directory
    """
    dataset_dir = os.path.join(base_dir, name)
    if clean and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    return os.path.join(dataset_dir, '')


@click.command()
@click.argument("name")
@click.option('--count', '-c', default=1000, help="Count of images to take. '1000' by default.")
@click.option('--base-dir', '-b', default=os.path.join(os.getcwd(), "dataset"),
              help="Base directory where dataset will be saved. './dataset' in this directory by default.")
@click.option('--clean', '-cl', default=False, help="Should clean existing data. 'False' by default.")
@click.option('--camera', '-ca', default=0, help="Camera input for CV2 lib. '0' by default.")
@click.option('--run-train', '-rt', default=True, help="Should run faces_train.py automatically? True default.")
def main(name, count, base_dir, clean, camera, run_train):
    """
    Basic script to create a dataset for face recognition app.

    :param name: Name of dataset
    :param count: Count of images to create
    :param base_dir: Base directory for storing the output
    :param clean: Should clean the folder if exists
    :param camera: Camera id for opencv lib
    :param run_train: Should run learning automatically
    """
    # cast strings to bool
    clean = bool(clean)
    run_train = bool(run_train)

    check_basedir(base_dir)
    dataset_dir = process_dataset_directory(base_dir, name, clean)
    create_dataset(dataset_dir, count, camera)
    if not run_train:
        print("Please run faces_train.py to let python learn from new created dataset.")
    else:
        train_faces()


if __name__ == '__main__':
    main()
