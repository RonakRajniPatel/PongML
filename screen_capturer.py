import numpy as np
import pyautogui
import imutils
import cv2


# this method takes a picture of the screen and saves it
def capture(index):
    image = pyautogui.screenshot()
    image = image[2000:3000][1000:2000][:]
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"pics/pic-{index}.png", image)


