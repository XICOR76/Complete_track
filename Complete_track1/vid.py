import cv2
import os

image_folder = "complete_track.v1i.yolov11/train/images"
output_video = "train_video.mp4"

images = sorted([img for img in os.listdir(image_folder)
                 if img.lower().endswith((".jpg", ".jpeg", ".png"))])

if not images:
    print("No images found!")
    exit()

# Read first image
first_frame = cv2.imread(os.path.join(image_folder, images[0]))

if first_frame is None:
    print("Error reading first image")
    exit()

height, width, _ = first_frame.shape

# Use a more compatible codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video, fourcc, 5, (width, height))

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # Resize to ensure consistent size
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()
cv2.destroyAllWindows()

print("Video created successfully:", output_video)