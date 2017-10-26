# USAGE
# python match.py --template cod_logo.png --images images

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import os


def find_object(image, template, category):
   
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    # cv2.imshow("Template", template)

# for imagePath in glob.glob(args["images"] + "/*.jpg"):

   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    print("find_object")

    # loop over the scales of the image
    # for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    for scale in [1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        print("maxVal: {}".format(maxVal))
        if maxVal > 0.1:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    if found:
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, category,(startX, startY - 5), font, 0.5, (0, 0, 255), 2)
   
    return image
   

def math_template(image):
    templates_dir = 'templates'

    for category in os.listdir(templates_dir):
        categ_dir = os.path.join(templates_dir, category)

        if os.path.isdir(categ_dir):
            for template in glob.glob(categ_dir + "/*.png"):
                template = cv2.imread(template)

                image = find_object(image, template, category)
   
    cv2.imshow("Image", image)


def process_video(videoPath):
    cap = cv2.VideoCapture(videoPath)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        math_template(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="Path to image where template will be matched")
    ap.add_argument("-v", "--video", help="Path to video")

    args = vars(ap.parse_args())

    imagePath = args["image"]
    videoPath = args["video"]

    if imagePath:
        image = cv2.imread(imagePath)
        math_template(image)
        cv2.waitKey(0)
    elif videoPath:
        process_video(videoPath)
    else:
        for image in glob.glob("images/*.jpg"):
            image = cv2.imread(image)
            math_template(image)
            cv2.waitKey(0)
   

if __name__ == '__main__':
    main()
