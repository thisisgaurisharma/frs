# import cv2

# # RTSP Stream URL (Replace with your camera details)
# rtsp_url = "rtsp://admin:admin123@192.168.1.204:554/avstream/channel=1/stream=1.sdp"

# # Open RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
# cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Lower buffer size
# cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # Skip unnecessary buffering

# if not cap.isOpened():
#     print("Error")
#     exit()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
   
#     cv2.imshow("video", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import numpy as np

# Load embeddings and names
embeddings = np.load("face_embeddings.npy", allow_pickle=True)
names = np.load("face_names.npy", allow_pickle=True)

# Display the data
print("Loaded Embeddings Shape:", embeddings.shape)
print("Loaded Names:", names)
