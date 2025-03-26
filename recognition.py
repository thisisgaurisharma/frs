import cv2
import numpy as np
import os
import faiss
from deepface import DeepFace

# Create a directory for saved faces if it doesn't exist
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")
    print("Directory 'saved_faces' created!")  # Debugging message

# FAISS index setup (512D for ArcFace)
embedding_dim = 512  # ArcFace model embedding dimension
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
known_embeddings = []  # Store embeddings
known_names = []  # Store corresponding names

# Load saved embeddings (if available)
if os.path.exists("face_embeddings.npy") and os.path.exists("face_names.npy"):
    known_embeddings = np.load("face_embeddings.npy", allow_pickle=True).tolist()
    known_names = np.load("face_names.npy", allow_pickle=True).tolist()
    index.add(np.array(known_embeddings, dtype=np.float32))

cap = cv2.VideoCapture(0)  # Use webcam (or RTSP link for IP cameras)

face_counter = 1  # Initialize a counter for saved faces

# Add known faces (you can add your face and another sample face manually)
known_faces = {
    "Johnathan Bailey": "baileyy.jpg",
    "Gauri Sharma": "testing.jpg"
}

# Add these faces to the FAISS index
for name, face_path in known_faces.items():
    face_crop = cv2.imread(face_path)
    embedding = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    known_embeddings.append(embedding.flatten())
    known_names.append(name)
    index.add(np.array([embedding.flatten()], dtype=np.float32))

# Loop to capture and process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Analyze frame for emotion and ethnicity
        results = DeepFace.analyze(frame_rgb, actions=['emotion', 'race'], detector_backend='retinaface', enforce_detection=False)
    except Exception as e:
        print("Detection Error:", e)
        results = []

    face_count = len(results)
    print(f"Detected {face_count} faces.")  # Debugging message

    # If faces are detected
    if face_count == 0:
        print("No faces detected in the current frame.")  # Debugging message
    else:
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green Box

            # Emotion and Ethnicity labels
            cv2.putText(frame, f"Emotion: {res['dominant_emotion']}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ethnicity: {res['dominant_race']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Get the Face Crop
            face_crop = frame[y:y+h, x:x+w]

            # Extract Embedding for Face Recognition
            try:
                embedding = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

                # FAISS Search for Recognition
                if len(known_embeddings) > 0:
                    D, I = index.search(embedding, 1)
                    label = known_names[I[0][0]] if D[0][0] < 0.6 else "Unknown"
                else:
                    label = "Unknown"

                # Display label on frame
                cv2.putText(frame, f"Label: {label}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # If Unknown, save the face & embedding
                if label == "Unknown":
                    print("Saving new face with embedding!")
                    face_path = f"saved_faces/face_{face_counter}.jpg"
                    cv2.imwrite(face_path, face_crop)

                    # Store embedding and name
                    known_embeddings.append(embedding.flatten())
                    known_names.append(face_path)
                    index.add(np.array([embedding.flatten()], dtype=np.float32))

                    # Save to disk
                    np.save("face_embeddings.npy", np.array(known_embeddings))
                    np.save("face_names.npy", np.array(known_names))

                    face_counter += 1  # Increment the counter for the next unknown face

            except Exception as e:
                print(f"Embedding Extraction Error: {e}")

    # Display Face Count
    cv2.putText(frame, f"Faces: {face_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show Webcam Feed
    cv2.imshow("Face Recognition (FAISS)", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
