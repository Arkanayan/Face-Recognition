import os
import re
import warnings
import scipy.misc
import cv2
import face_recognition
from PIL import Image
import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images-dir", help="training image dir")
parser.add_argument("-v", "--video", help="video to recognize faces on")
parser.add_argument("-o", "--output-csv", help="Ouput csv file")
args = vars(parser.parse_args())

if args.get("images_dir", None) is None and os.path.exists(args.get("images_dir", None)):
    print("Please check the path to images folder")
    exit()
if args.get("video", None) is None and os.path.isfile(args.get("video", None)):
    print("Please check the path to video")
    exit()
if args.get("output_csv", None) is None:
    print("You haven't specified an output csv file. Nothing will be written.")



# Helper functions


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def test_image(image_to_check, known_names, known_face_encodings):
    # unknown_image = face_recognition.load_image_file(image_to_check)
    unknown_image = image_to_check
    # Scale down the image to make it run faster
    if unknown_image.shape[1] > 1600:
        scale_factor = 1600 / unknown_image.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unknown_image = scipy.misc.imresize(unknown_image, scale_factor)
    face_locations = face_recognition.face_locations(unknown_image)
    unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    result = []
    for unknown_encoding in unknown_encodings:
        result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

    result_encoding = []
    for nameIndex, is_match in enumerate(result):
        if is_match:
            result_encoding.append(known_names[nameIndex])

    return result_encoding

def map_file_pattern_to_label(labels_with_pattern, labels_list):
    """
    Map file name pattern to full label
    :param labels_with_pattern: dict : { "file_name_pattern": "full_label" }
    :param labels_list: list : list of labels of file names got from test_image()
    :return: list of full labels
    """
    result_list = []
    for key, label in labels_with_pattern.items():
        for img_labels in labels_list:
            if str(key).lower() in str(img_labels).lower():
                if str(label) not in result_list:
                    result_list.append(str(label))
                # continue
    # result_list = [label for key, label in labels_with_pattern if str(key).lower() in labels_list]
    return result_list

cap = cv2.VideoCapture(args["video"])

training_encodings = []
training_labels = []
for file in image_files_in_folder(args['images_dir']):
    basename = os.path.splitext(os.path.basename(file))[0]
    img = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 1:
        print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

    if len(encodings) == 0:
        print("WARNING: No faces found in {}. Ignoring file.".format(file))
    if len(encodings):
        training_labels.append(basename)
        training_encodings.append(encodings[0])

# known_image = face_recognition.load_image_file(args["image"])
# known_face_encoding = face_recognition.face_encodings(known_image)[0]

ret, firstFrame = cap.read()

csvfile = None
csvwriter = None
if args.get("output_csv", None) is not None:
    csvfile = open(args.get("output_csv"), 'w')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

ret, firstFrame = cap.read()
frameRate = cap.get(cv2.CAP_PROP_FPS)

while ret:
    curr_frame = cap.get(1)

    ret, frame = cap.read()
    # face_locations = face_recognition.face_locations(frame)
    # print(face_locations)
    # cv2.putText(frame, "{}".format(len(face_locations)), (10, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # for face_location in face_locations:
    #     top, right, bottom, left = face_location
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(frame, "0", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # detected_face_encodings = face_recognition.face_encodings(frame)
    # for encodings in detected_face_encodings:
    #     results = face_recognition.compare_faces([known_face_encoding], detected_face_encodings)
    #     for result in results:
    #         if result == True:
    #             cv2.putText(frame, "1", (10, 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    result = test_image(frame, training_labels, training_encodings)
    labels = map_file_pattern_to_label({"shah": "Shah Rukh khan", "kapil": "Kapil Sharma"}, result)
    curr_time = curr_frame / frameRate
    print("Time: {} faces: {}".format(curr_time, labels))
    if csvwriter:
        csvwriter.writerow([curr_time, labels])
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
if csvfile:
    csvfile.close()
cap.release()
cv2.destroyAllWindows()
