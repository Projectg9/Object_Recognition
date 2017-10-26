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
    cv2.imshow("Template", template)

# for imagePath in glob.glob(args["images"] + "/*.jpg"):

   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
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
        if maxVal > 0.9:
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
        cv2.imshow("Image", image)
        cv2.waitKey(0)
   

def math_template(image):
    templates_dir = 'templates'

    for category in os.listdir(templates_dir):
        categ_dir = os.path.join(templates_dir, category)

        if os.path.isdir(categ_dir):
            for template in glob.glob(categ_dir + "/*.png"):
                template = cv2.imread(template)

                find_object(image, template, category)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="Path to image where template will be matched")
    args = vars(ap.parse_args())

    imagePath = args["image"]
    if imagePath:
        image = cv2.imread(imagePath)
        math_template(image)
    else:
        for image in glob.glob("images/*.jpg"):
            image = cv2.imread(image)
            math_template(image)
   

if __name__ == '__main__':
    main()

