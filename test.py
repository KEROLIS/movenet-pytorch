from image_demo import draw_keypoints_on_image
import cv2

image = draw_keypoints_on_image(image="/mnt/d587d2ae-9267-4fad-a5cc-c62c5d676c66/pulses.ai/movenet_pytorch/images/frisbee.jpg",model_name="movenet_thunder",conf_threshold=.3)

cv2.imshow("test",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
