import cv2
import mediapipe as mp
import numpy as np
import threading

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Camera and Canvas
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)
use_blackboard = False
show_video_canvas = True  # Flag to toggle between canvas and blackboard

previous_point = None
current_color = (255, 255, 255)
brush_thickness = 5
erase_thickness = 50

# Define Buttons
color_buttons = {
    'BLUE': ((50, 50), (150, 150), (255, 0, 0)),
    'GREEN': ((170, 50), (270, 150), (0, 255, 0)),
    'RED': ((290, 50), (390, 150), (0, 0, 255)),
    'YELLOW': ((410, 50), (510, 150), (0, 255, 255))
}
blackboard_button = ((530, 50), (680, 150))
exit_button = ((1080, 600), (1250, 680))
clear_button = ((880, 600), (1050, 680))

# Function to open a separate blackboard window
def open_blackboard():
    global blackboard, use_blackboard, show_video_canvas, previous_point
    while use_blackboard:
        frame_copy = blackboard.copy()

        # Draw Color Buttons and Clear Button on Blackboard
        for name, (top_left, bottom_right, color) in color_buttons.items():
            cv2.rectangle(frame_copy, top_left, bottom_right, color, -1)
            cv2.putText(frame_copy, name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame_copy, clear_button[0], clear_button[1], (0, 0, 0), -1)
        cv2.putText(frame_copy, 'CLEAR', (clear_button[0][0] + 10, clear_button[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Draw Exit Buttons on Blackboard
        cv2.rectangle(frame_copy, exit_button[0], exit_button[1], (0, 0, 0), -1)
        cv2.putText(frame_copy, 'EXIT', (exit_button[0][0] + 20, exit_button[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        

        # Draw Hand Landmarks on Blackboard
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Gesture Drawing on Blackboard
        if previous_point is not None:
            cv2.line(frame_copy, previous_point, previous_point, current_color, brush_thickness)  # Draw line continuously

        cv2.imshow("Blackboard", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'Q' key press to close the blackboard
            use_blackboard = False
            cv2.destroyWindow("Blackboard")
            break


        # Check for Exit Button click
        if exit_button[0][0] < index_tip[0] < exit_button[1][0] and exit_button[0][1] < index_tip[1] < exit_button[1][1]:
            use_blackboard = False
            cv2.destroyWindow("Blackboard")
            break

def fingers_up(hand_landmarks):
    fingers = []
    landmarks = hand_landmarks.landmark
    fingers.append(landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x)
    fingers += [
        landmarks[tip].y < landmarks[tip - 2].y
        for tip in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]
    ]
    return fingers

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw Buttons on Main Canvas
    for name, (top_left, bottom_right, color) in color_buttons.items():
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.putText(frame, name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, blackboard_button[0], blackboard_button[1], (100, 100, 100), -1)
    cv2.putText(frame, 'BLACKBOARD', (blackboard_button[0][0] + 5, blackboard_button[0][1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, clear_button[0], clear_button[1], (0, 0, 0), -1)
    cv2.putText(frame, 'CLEAR', (clear_button[0][0] + 10, clear_button[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.rectangle(frame, exit_button[0], exit_button[1], (0, 0, 0), -1)
    cv2.putText(frame, 'EXIT', (exit_button[0][0] + 20, exit_button[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                         int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
            fingers = fingers_up(hand_landmarks)

            # Color Selection on Main Canvas
            for name, (top_left, bottom_right, color) in color_buttons.items():
                if top_left[0] < index_tip[0] < bottom_right[0] and top_left[1] < index_tip[1] < bottom_right[1]:
                    current_color = color

            # Blackboard Toggle
            if blackboard_button[0][0] < index_tip[0] < blackboard_button[1][0] and blackboard_button[0][1] < index_tip[1] < blackboard_button[1][1]:
                if not use_blackboard:
                    use_blackboard = True
                    threading.Thread(target=open_blackboard).start()

            # Left Hand Eraser
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > 0.5 and fingers.count(True) == 0:
                cv2.circle(canvas if not use_blackboard else blackboard, index_tip, erase_thickness, (0, 0, 0), -1)
            elif fingers[1] and not any(fingers[2:]):
                if previous_point is None:
                    previous_point = index_tip
                else:
                    cv2.line(canvas if not use_blackboard else blackboard, previous_point, index_tip, current_color, brush_thickness)
                    previous_point = index_tip
            else:
                previous_point = None

            # Clear Button on Main Canvas
            if clear_button[0][0] < index_tip[0] < clear_button[1][0] and clear_button[0][1] < index_tip[1] < clear_button[1][1]:
                canvas.fill(0)
                blackboard.fill(0)
            # Exit Button
            if exit_button[0][0] < index_tip[0] < exit_button[1][0] and exit_button[0][1] < index_tip[1] < exit_button[1][1]:
                cap.release()
                cv2.destroyAllWindows()
                exit()

    if show_video_canvas:
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Video Canvas", combined)
    else:
        cv2.imshow("Virtual Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()