# import cv2
# from deepface import DeepFace

# # Open webcam (0 is usually the default camera)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ Error: Unable to open webcam.")
#     exit()

# # Set a frame width and height for resizing
# frame_width = 640
# frame_height = 480

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("⚠️ Failed to grab frame. Retrying...")
#         continue  # Skip frame if not captured

#     # Resize the frame for faster processing
#     frame_resized = cv2.resize(frame, (frame_width, frame_height))

#     # Convert frame to RGB (DeepFace requires RGB input)
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

#     try:
#         # Perform face detection and analysis using MTCNN for better accuracy
#         results = DeepFace.analyze(
#             frame_rgb, actions=['emotion', 'race'], 
#             detector_backend='mtcnn',  # Use MTCNN for better accuracy
#             enforce_detection=False
#         )
#     except Exception as e:
#         print("🚨 Detection Error:", e)
#         results = []

#     # Draw results on the frame
#     if results:  # Check if results are not empty
#         for res in results:
#             if 'region' in res and res['region']:  # Ensure region exists
#                 x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']

#                 # Draw a bounding box
#                 cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green Box

#                 # Extract detected attributes
#                 emotion = res.get('dominant_emotion', 'Unknown')
#                 ethnicity = res.get('dominant_race', 'Unknown')

#                 # Display labels on the frame
#                 cv2.putText(frame_resized, f"Emotion: {emotion}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#                 cv2.putText(frame_resized, f"Ethnicity: {ethnicity}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#                 # Print results to the terminal
#                 print(f"Detected Face - Emotion: {emotion}, Ethnicity: {ethnicity}")

#         # Display the count of detected faces
#         face_count = len(results)
#         cv2.putText(frame_resized, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#     # Display the frame
#     cv2.imshow("Webcam Multiple Face Detection", frame_resized)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# from deepface import DeepFace
# from threading import Thread
# import queue

# # RTSP Stream URL (Replace with your actual RTSP link)
# RTSP_URL = "rtsp://admin:123456@192.168.1.205:554/avstream/channel=1/stream=0.sdp"

# # Initialize a queue to hold frames
# frame_queue = queue.Queue(maxsize=10)

# # Function to capture frames from the RTSP stream
# def capture_frames():
#     cap = cv2.VideoCapture(RTSP_URL)
#     if not cap.isOpened():
#         print("❌ Error: Unable to open RTSP stream. Check the URL.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Failed to grab frame. Retrying...")
#             continue  # Skip frame if not captured

#         if not frame_queue.full():
#             frame_queue.put(frame)  # Add frame to the queue

#     cap.release()

# # Function to process frames from the queue
# def process_frames():
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()  # Get frame from the queue
#             frame_resized = cv2.resize(frame, (640, 480))  # Resize frame
#             frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB

#             try:
#                 results = DeepFace.analyze(
#                     frame_rgb, actions=['emotion', 'race'],
#                     detector_backend='mtcnn',
#                     enforce_detection=False
#                 )
#             except Exception as e:
#                 print("🚨 Detection Error:", e)
#                 results = []

#             # Draw results on the frame
#             if results:
#                 for res in results:
#                     if 'region' in res and res['region']:
#                         x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
#                         cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green Box
#                         emotion = res.get('dominant_emotion', 'Unknown')
#                         ethnicity = res.get('dominant_race', 'Unknown')
#                         cv2.putText(frame_resized, f"Emotion: {emotion}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#                         cv2.putText(frame_resized, f"Ethnicity: {ethnicity}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#             # Display the processed frame
#             cv2.imshow("RTSP Stream Multiple Face Detection", frame_resized)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

# # Start threads for capturing and processing frames
# capture_thread = Thread(target=capture_frames)
# process_thread = Thread(target=process_frames)

# capture_thread.start()
# process_thread.start()

# capture_thread.join()
# process_thread.join()

# cv2.destroyAllWindows()