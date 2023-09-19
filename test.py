import cv2
import re

# img = cv2.imread("cat.png", 1)
# cv2.imshow("1", img)
# cv2.waitKey()
a = './weight/640_模型.onnx'
b = int(re.split(r'[/_]', a)[-2])
print(b)