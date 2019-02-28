# author Jiri Volprecht
import os
import pickle
from collections import defaultdict

import click
import cv2
import pandas as pd
from datetime import datetime
from glob import glob


def validate_files():
    """
    Check if files created from faces_train.py script
    """
    recognizer_f = glob("./recognizers/*.yml")
    pickle_f = glob("./pickles/*.pickle")
    if not len(recognizer_f) or not len(pickle_f):
        raise Exception("Missing files for recognizing people. Please create a dataset and run faces_train.py first.")


def validate_folder_structure():
    """
    Check if folder structure is correct
    """
    if not os.path.isdir("./cascades/data/") or \
            not os.path.isdir("./recognizers") or \
            not os.path.isdir("./pickles") or \
            not os.path.isdir("./reports"):
        raise Exception("Missing compulsory folder structure. Please do git checkout.")


def process_data(people_logger):
    """
    Creates a dictionary of DataFrames from the log and write the results to csv file.
    :param people_logger: input data from face recognition
    :return: dictionary of DataFrames
    """
    # create dictionary DataFrames with name as a key and times
    ppl_logger_df = {key: pd.DataFrame(people_logger[key]) for key in people_logger.keys()}
    # save data to csv
    [ppl_logger_df[key].to_csv(os.path.join(os.getcwd(), "reports", f"{key}.csv")) for key in ppl_logger_df.keys()]
    return ppl_logger_df


@click.command()
@click.option('--camera', '-c', default=0, help="Camera source. '0' by default.")
@click.option('--report', '-r', default=True, help="Should creates report of people?. 'True' by default.")
@click.option('--capture-unknown', '-cu', default=True, help="Should creates report of people?. 'True' by default.")
def main(camera, report, capture_unknown):
    # cast strings to bool
    report = bool(report)
    capture_unknown = bool(capture_unknown)

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./recognizers/face-trainner.yml")
    active_people = set()
    people_on_cam = set()
    people_logger = defaultdict(list)

    with open("pickles/face-labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        # switching key and val in dict
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(camera)

    unknown_dir = os.path.join(os.getcwd(), "dataset", "unknown", "")

    while True:
        ret, frame = cap.read()

        # transform to grey scale to use face cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        people_on_cam.clear()
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            # save face image
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            # recognizer
            id_, conf = recognizer.predict(roi_gray)

            if 40 <= conf <= 85:
                name = labels[id_]
            else:
                # if unknown person captured, take a picture of him and save it to the folder
                name = "Unknown"

                if capture_unknown:
                    file_base_name = datetime.now().timestamp()
                    cv2.imwrite(f"{unknown_dir}{file_base_name}.png", frame)
                    cv2.imwrite(f"{unknown_dir}{file_base_name}_face.png", roi_color)

            if name not in active_people:
                people_logger[name].append({'start': datetime.now()})
                if capture_unknown and name == "Unknown":
                    people_logger[name][-1]['photo'] = file_base_name
                active_people.add(name)
            people_on_cam.add(name)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            # draw rectangle
            color = (255, 0, 0)  # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # check who disappeared
        print("on cam: ", people_on_cam)
        print("active: ", active_people)
        missing = active_people - people_on_cam

        for n in missing:
            people_logger[n][-1]["end"] = datetime.now()
            active_people.remove(n)

        if report:
            process_data(people_logger)

        # display result
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    validate_folder_structure()
    validate_files()
    main()
