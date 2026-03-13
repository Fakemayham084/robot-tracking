import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]

MODEL_FILES = {'pose': 'pose_landmarker_heavy.task', 'hand': 'hand_landmarker.task', 'face': 'face_landmarker.task'}
selected_id = -1 

def create_detector(model_path, task_type):
    base_options = python.BaseOptions(model_asset_path=model_path)
    if task_type == 'pose':
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=4)
        return vision.PoseLandmarker.create_from_options(options)
    elif task_type == 'hand':
        options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=6)
        return vision.HandLandmarker.create_from_options(options)
    elif task_type == 'face':
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces=4)
        return vision.FaceLandmarker.create_from_options(options)

def select_person(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for p_id, box in param['boxes'].items():
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                selected_id = p_id
                print(f"Locked onto Person {selected_id}")

pose_det = create_detector(MODEL_FILES['pose'], 'pose')
hand_det = create_detector(MODEL_FILES['hand'], 'hand')
face_det = create_detector(MODEL_FILES['face'], 'face')

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Tracker')
tracking_context = {'boxes': {}}
cv2.setMouseCallback('Interactive Tracker', select_person, tracking_context)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    curr_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    pose_res = pose_det.detect_for_video(mp_img, curr_time)
    hand_res = hand_det.detect_for_video(mp_img, curr_time)
    face_res = face_det.detect_for_video(mp_img, curr_time)

    temp_boxes = {}
    id_found = False 

    if pose_res.pose_landmarks:
        for idx, landmarks in enumerate(pose_res.pose_landmarks):
            pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
            x_min, y_min = min(p[0] for p in pts), min(p[1] for p in pts)
            x_max, y_max = max(p[0] for p in pts), max(p[1] for p in pts)
            temp_boxes[idx] = (x_min, y_min, x_max, y_max)

            if idx == selected_id:
                id_found = True
                color = (0, 0, 255) 
            else:
                color = (0, 255, 0) 

            for connection in POSE_CONN:
                cv2.line(frame, pts[connection[0]], pts[connection[1]], color, 2)
            for pt in pts: 
                cv2.circle(frame, pt, 3, (255, 255, 255), -1)
            
            cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), color, 2)
            cv2.putText(frame, f"ID: {idx}", (x_min, y_min-15), 0, 0.6, color, 2)

    if selected_id != -1 and not id_found:
        print(f"Lost track of Person {selected_id}. Unselecting...")
        selected_id = -1

    if hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            h_pts = [(int(l.x * w), int(l.y * h)) for l in hand]
            for conn in HAND_CONN:
                cv2.line(frame, h_pts[conn[0]], h_pts[conn[1]], (255, 0, 255), 1)
            for pt in h_pts: cv2.circle(frame, pt, 2, (255, 255, 255), -1)

    if face_res.face_landmarks:
        for face in face_res.face_landmarks:
            for l in face:
                cv2.circle(frame, (int(l.x * w), int(l.y * h)), 1, (180, 180, 180), -1)

    status_text = f"Locked: {selected_id}" if selected_id != -1 else "Locked: NONE"
    cv2.putText(frame, status_text, (20, 40), 0, 0.8, (0, 255, 255), 2)

    tracking_context['boxes'] = temp_boxes
    cv2.imshow('Interactive Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()