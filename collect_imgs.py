import os
import cv2

# Directory where data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 100

# Initialize the video capture object with the correct index for your camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    # Create a directory for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret or frame is None:
            print("Error: Failed to capture frame. Please check the camera.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for the user to press 'Q' to start capturing images
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret or frame is None:
            print("Error: Failed to capture frame. Please check the camera.")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame to the appropriate directory
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
