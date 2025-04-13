import cv2
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

dataset_path = (r"C:\Users\loltr\ASLImages") #"C:\Users\Amy\Desktop\content"
IMG_SIZE = 200
data = []
labels = []


def remove_background(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #It's a bit noisy lets reduce it
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #Sift to find key points
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(edges, None)
    # this is for debugging it highlights what it is lookin at
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if keypoints:
        #
        hull = cv2.convexHull(np.array([kp.pt for kp in keypoints], dtype=np.int32))

        
        hand_mask = np.zeros_like(gray)
        cv2.fillConvexPoly(hand_mask, hull, 255)
        
        return frame_with_keypoints, hand_mask

    # Return the frame with keypoints and an empty mask if no keypoints are detected
    return frame_with_keypoints, np.zeros_like(gray)

# Load and preprocess the dataset
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path not found: {dataset_path}")
else:
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)

        # Check if it's an image file
        if not img_name.lower().endswith(('.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Could not read image: {img_path}")
                continue

            # Extract label from image name
            sign = img_name.split('_')[1]
            if sign.isalpha():
                label = ord(sign.upper()) - ord('A')
            elif sign.isdigit():
                label = int(sign) + 26
            else:
                print(f"Unexpected sign format: {sign}")
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)

        except Exception as e:
            print(f"⚠️ Error processing image {img_name}: {e}")

# Convert to numpy arrays and preprocess
if data:
    data = np.array(data, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    labels = np.array(labels)

    # One-hot encode the labels manually
    def one_hot_encode(labels, num_classes):
        one_hot_labels = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        return one_hot_labels

    labels = one_hot_encode(labels, 36)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train.reshape(len(x_train), -1), np.argmax(y_train, axis=1))  # Flatten images for DT

    # Evaluate the classifier
    y_pred = clf.predict(x_test.reshape(len(x_test), -1))  # Flatten test images
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
else:
    print("⚠️ Error: No images found or processed successfully.")


def asl_recognition(frame):
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.reshape(1, -1) / 255.0  # Flatten and normalize the image
    prediction = clf.predict(img_flattened)
    if prediction[0] < 26:
        translated_word = chr(prediction[0] + ord('A'))  # A-Z
    else:
        translated_word = str(prediction[0] - 26)  # 0-9
    return frame, translated_word


# Start Xvfb in the main thread and wait for it to start
os.system('Xvfb :1 -screen 0 1024x768x24 &')
os.environ['DISPLAY'] = ':1'  # Set DISPLAY environment variable
import time
time.sleep(5)  # Give Xvfb some time to start and create display :1

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open the camera")
    exit()

# Takes in live data from webcam to send to the gamers
def get_live_data(label, word_label):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("⚠️ Cannot open the camera")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Error: Failed to capture frame.")
            break

        asl_frame, translated_word = asl_recognition(frame)

        # Display the frame and predicted word
        cv2image = cv2.cvtColor(asl_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=img)

        label.configure(image=img)
        label.image = img

        word_label.configure(text=f"Predicted Word: {translated_word}")

        cv2.waitKey(1)

def user_interface():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title("ASL Translator")
    root.geometry("800x600")

    title_label = ctk.CTkLabel(root, text="We Present, The ASL Translator!", font=("Times New Roman", 20))
    title_label.pack(pady=20)

    webcam_label = ctk.CTkLabel(root)
    webcam_label.pack(pady=20)

    word_label = ctk.CTkLabel(root, text="Predicted Word: ", font=("Arial", 18))
    word_label.pack(pady=20)

    threading.Thread(target=get_live_data, args=(webcam_label, word_label), daemon=True).start()

    root.mainloop()

user_interface()
