import numpy as np
import pyautogui
import imutils
import cv2.cv2 as cv2


def capture(index):
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"pic-{index}.png", image)


