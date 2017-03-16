import cv2
import face_recognition
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="training image")
parser.add_argument("-v", "--video", help="video to recognize faces on")
args = vars(parser.parse_args())

if args.get("image", None) is None:
    print("Please check the path to image")
    exit()
if args.get("video", None) is None:
    print("Please check the path to video")
    exit()

cap = cv2.VideoCapture(args["video"])

known_image = face_recognition.load_image_file(args["image"])
known_face_encoding = face_recognition.face_encodings(known_image)[0]

ret, firstFrame = cap.read()

while ret:

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
    detected_face_encodings = face_recognition.face_encodings(frame)
    for encodings in detected_face_encodings:
        results = face_recognition.compare_faces([known_face_encoding], detected_face_encodings)
        for result in results:
            if result == True:
                cv2.putText(frame, "1", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
