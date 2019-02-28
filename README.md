Live face-recognition tool with logging to csv file

Recommended python version is 3.6.

# Installation
I strongly recommend to create a virtualenv by command `python -m venv virtual`. Then activate it and install requirements from requirements.txt file. There is a problematic library to install called dlib. If the pip install failed, please follow the instructions here: https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/

## Creating a dataset
To let python recognize faces you have to create a dataset. It's recommended to use at least 1000 pics and every 
person should have similar count of pictures. More pictures provided by you = better recognition. You have two 
options how to create it.

### Manual creation of dataset
You can find your own photos just with you and copy them to created folder in `dataset/<your_name>`. Folder name will be 
the label of the recognized person in video.

### Automatic creation of dataset
No one probably find such amount of pictures. That's why `create_dataset.py` script is here. It will use your webcam or 
any other source of video provided by parameter to create a dataset for you. For example for dataset for George you can 
run it like this:

`python create_dataset.py george` 

This command will take 1000 pictures of you from your webcam. Please be patient, it will take some time. The script is
easily configurable. To see what else you can change run:

`python create_dataset.py --help`


## Training recognition

When you already have dataset created you have to let python to learn from these files. If you used **create_dataset** 
with default settings or you specified **-rt True** option, this happened automatically. Otherwise you have to run 
`python faces-train.py` script.

## Run the program

Now everything is set up and you can run the program

`python faces.py`

It will open the window with the video from your webcam and if it recognize the person from the dataset it will show the label.
Every recognition is written down to csv file with the times. In case unknown person is in there it will save also the 
picture of him/her and save it to dataset/unknown folder. When you want to exit the program just press "q" on your keyboard.
