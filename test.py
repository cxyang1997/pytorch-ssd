from vision.ssd.data_preprocessing import PredictionTransform
import cv2


image_path = 'gun.jpg'
orig_image = cv2.imread(image_path)
cur_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

trans = PredictionTransform(300, [123, 117, 104], 1.0)
image = trans(cur_image)
images = image.unsqueeze(0)


