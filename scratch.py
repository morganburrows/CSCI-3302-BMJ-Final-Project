
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')

pil_image = Image.open("test_image.png").convert("L")
# pil_image.show()
final_image = pil_image.resize((550, 550), Image.ANTIALIAS)
image_array = np.array(final_image, "uint8")
#final_image.show()
faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)  # higher scale facter might increase accuracy
for (x, y, w, h) in faces:
    print(x, y, w, h)
    #roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end) region of interest
    roi_color = image_array[y:y + h, x:x + w]
    #img_item = "my-image.png"
    #cv2.imwrite(img_item, roi_gray)

    # recognizer
    id_, conf = recognizer.predict(roi_gray)
    if conf >= 15:  # and conf <= 85:
        print(id_)
        print(labels[id_])