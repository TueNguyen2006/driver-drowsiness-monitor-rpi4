import cv2 
import mediapipe as mp
import math
import numpy as np 
import torch
import pickle
import warnings
import gc
import threading
import state
import config as cfg
import time
from hardware import HardwareController

detect = False
hardware_controller = None

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    p1 = np.array(p1)  # Chuyển đổi nếu p1 là list hoặc tuple
    p2 = np.array(p2)  # Chuyển đổi nếu p2 là list hoặc tuple
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def face_width(landmarks):
    return distance(landmarks[162], landmarks[389])

def face_height(landmarks):
    return distance(landmarks[10], landmarks[4])

def mouth_height(landmarks):
    return distance(landmarks[14], landmarks[13])

def mouth_width(landmarks):
    return distance(landmarks[69], landmarks[291])


def classify_mouth_behavior_frames(
    landmarks_series: list,
    baseline_dh: float,
    # Số frame tối thiểu để coi là ngáp hoặc tối đa cho nói
    yawn_frames_thr: int = 24,      # ví dụ ngáp cần ≥24 frame (~0.8s tại 30fps)
    speak_frames_thr: int = 12,     # ví dụ nói ≤12 frame (~0.4s)
    symmetry_frames_thr: int = 6,   # độ lệch rise/fall ≤6 frame (~0.2s)
    # Ngưỡng vertical opening
    yawn_dv_thr: float = 0.7,
    speak_dv_thr: float = 0.6,
    # Số peak tối đa và tần số peak
    peaks_v_thr: int = 2,
    peaks_h_freq_thr: float = 2.0
) -> str:
    """
    Phân loại 'yawn', 'speak' hoặc 'none' dựa vào số lượng frames và landmarks.
    landmarks_series: list of MediaPipe landmarks mỗi frame trong segment
    baseline_dh: khoảng cách khóe môi nền
    """
    n = len(landmarks_series)
    if n < 2:
        return 'none'

    # Tính DV và DH normalized
    dv = []
    dh = []
    for lm in landmarks_series:
        up = np.array([lm[13].x, lm[13].y])
        dn = np.array([lm[14].x, lm[14].y])
        left = np.array([lm[61].x, lm[61].y])
        right = np.array([lm[291].x, lm[291].y])
        dv.append(np.linalg.norm(up - dn) / baseline_dh)
        dh.append(np.linalg.norm(left - right) / baseline_dh)
    dv = np.array(dv)
    dh = np.array(dh)

    # Tính các đặc trưng
    dv_peak = dv.max()
    dh_min = dh.min()
    idx_peak = dv.argmax()
    rise = idx_peak
    fall = n - idx_peak - 1
    symmetry = abs(rise - fall)

    deriv_v = np.diff(dv)
    sign_v = np.sign(deriv_v)
    peaks_v = int(np.sum((sign_v[:-1] > 0) & (sign_v[1:] < 0)))

    deriv_h = np.diff(dh)
    sign_h = np.sign(deriv_h)
    peaks_h = int(np.sum((sign_h[:-1] > 0) & (sign_h[1:] < 0)))
    freq_h = peaks_h / (n / (yawn_frames_thr / speak_frames_thr + 1))  # ước lượng tần số

    # Rule-based classification
    if (
        dv_peak >= yawn_dv_thr and
        dh_min <= 0.6 and
        n >= yawn_frames_thr and
        peaks_v <= peaks_v_thr and
        symmetry <= symmetry_frames_thr
    ):
        return 'yawn'
    if (
        n <= speak_frames_thr and
        freq_h >= peaks_h_freq_thr and
        dv_peak < speak_dv_thr
    ):
        return 'speak'
    return 'none'


def normalize_test(poses_array):
    # poses_array: array với các giá trị theo thứ tự ['nose_x', 'nose_y', ..., 'mouth_right_y']
    normalized_array = poses_array.copy()
    # Indices của các feature trong array
    
    for dim_idx in [0, 1]:  # X và Y tương ứng là chỉ số chẵn/lẻ
        # Centering around the nose
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = poses_array[feature_idx] - poses_array[dim_idx]  # Trừ giá trị nose_x hoặc nose_y
        
        # Scaling
        diff = poses_array[12 + dim_idx] - poses_array[4 + dim_idx]  # mouth_right_dim - left_eye_dim
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = normalized_array[feature_idx] / diff

    # Tạo array 2D để phù hợp với đầu vào của mô hình
    face_features_array = [normalized_array]
    return face_features_array


def head_pose(face_features):
    
    pitch_pred, yaw_pred, roll_pred = 0, 0, 0

    face_features_normalized = normalize_test(face_features)
        
    # Dự đoán các giá trị pitch, yaw, roll
    pitch_pred, yaw_pred, roll_pred = model_head_pose.predict(face_features_normalized).ravel()

    return pitch_pred, yaw_pred, roll_pred

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img



def run_face_mp(image, height, width, draw_face = True):
    global alert, running, running_inference, arduino, servo_delta_x, servo_delta_y, current_servo_x, current_servo_y, detect
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291
    face_features = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for idx, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y]) # saving normalized landmark positions
            if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                face_features.append(data_point.x)
                face_features.append(data_point.y)
        
        landmarks_positions = np.array(landmarks_positions)
        pitch_pred, yaw_pred, roll_pred = head_pose(face_features)

        landmarks_positions[:, 0] *= width
        landmarks_positions[:, 1] *= height

        # draw face mesh over image
        if draw_face:
            for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
        
        Nose_x = int(landmarks_positions[NOSE][0])
        Nose_y = int(landmarks_positions[NOSE][1])

        image = draw_axes(image, pitch_pred, yaw_pred, roll_pred, Nose_x, Nose_y)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
        detect = True
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000
        pitch_pred, yaw_pred, roll_pred = 0, 0, 0
        detect = False
    results = None
    return ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image

def calibrate(calib_frame_count=100, frames_start = 0):
    state.mode = "calibrate"

    ears = []
    mars = []
    pucs = []
    moes = []
    pitch_preds = []
    yaw_preds = []
    roll_preds = []

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(3, width)
    # cap.set(4, height)
    frames = 0
    
    while state.running:
        success, image = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        frames +=1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image = run_face_mp(image, height=height, width=width)
        if ear != -1000 and frames > frames_start:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            pitch_preds.append(pitch_pred)
            yaw_preds.append(yaw_pred)
            roll_preds.append(roll_pred)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if frames >= frames_start + calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    state.mode = "idle"

    if not state.running or not ears:
        return None

    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)

    pitch_preds = np.array(pitch_preds)
    pitch_mean_zscore = pitch_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    yaw_preds = np.array(yaw_preds)
    yaw_mean_zscore = yaw_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    roll_preds = np.array(roll_preds)
    roll_mean_zscore = roll_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()], \
        pitch_mean_zscore, yaw_mean_zscore, roll_mean_zscore


def get_classification(input_data):
    ''' Perform classification over the facial  features.
    :param input_data: List of facial features for 20 frames
    :return: Alert / Drowsy state prediction
    '''

    with torch.no_grad():
        model_input = [input_data[i:i+5] for i in range(0, 10, 3)]
        model_input = torch.FloatTensor(np.array(model_input))
        preds = model(model_input)
        preds = (preds > 0.5).int().cpu().numpy()
    return int(preds.sum() >= 3)



def infer(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred_norm, yaw_pred_norm, roll_pred_norm, ras_obj=None):
    ''' Perform inference.
    :param ears_norm: Normalization values for eye feature
    :param mars_norm: Normalization values for mouth feature
    :param pucs_norm: Normalization values for pupil feature
    :param moes_norm: Normalization values for mouth over eye feature. 
    :param 
    '''
    global detect
    state.mode = "infer"
    state.request_recalibrate = False

    def threshhold_EAR_MAR(ear_main, mar_main, EAR_threshold = -2, MAR_threshold = 30):
        return (ear_main < EAR_threshold or mar_main > MAR_threshold)
    
    def out_of_camera():
        nonlocal out_of_camera_announced
        if hardware_controller is not None and not out_of_camera_announced:
            hardware_controller.speak_async("Vui lòng nhìn về phía trước", dedupe_window=8.0)
            out_of_camera_announced = True

    def on_warning_drowsy():
        nonlocal drowsy_condition_started_at, alert_release_at
        now = time.monotonic()
        if drowsy_condition_started_at is None:
            drowsy_condition_started_at = now
        if now - drowsy_condition_started_at >= cfg.ALERT_STABLE_SECONDS:
            state.alert = True
            alert_release_at = now + cfg.ALERT_HOLD_SECONDS

    def off_warning_drowsy():
        state.alert = False

    def open_camera():
        new_cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        if not new_cap.isOpened():
            if hardware_controller is not None:
                hardware_controller.speak_async("Không mở được camera", dedupe_window=10.0)
            return None, 640, 480
        cam_width = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        cam_height = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        return new_cap, cam_width, cam_height

    def reset_inference_buffers():
        nonlocal ear_main, mar_main, puc_main, moe_main
        nonlocal pitch_main, yaw_main, roll_main
        nonlocal head, lost_focus_count, pitch_count, yaw_count
        nonlocal triggered, count_EAR, count_MAR
        nonlocal max_yaw, min_yaw, lost_focus, drowsy
        nonlocal label, input_data, frame_before_run, count_decision
        nonlocal alert_release_at, drowsy_condition_started_at, missed_count, out_of_camera_announced

        ear_main = -1000
        mar_main = -1000
        puc_main = -1000
        moe_main = -1000
        pitch_main = 0
        yaw_main = 0
        roll_main = 0
        head = 0
        lost_focus_count = 0
        pitch_count = 0
        yaw_count = 0
        triggered = False
        count_EAR = 0
        count_MAR = 0
        max_yaw = 40
        min_yaw = -40
        lost_focus = False
        drowsy = False
        label = None
        input_data = []
        frame_before_run = 0
        count_decision = 0
        alert_release_at = 0.0
        drowsy_condition_started_at = None
        missed_count = 0
        out_of_camera_announced = False
        state.alert = False

    ear_main = -1000
    mar_main = -1000
    puc_main = -1000
    moe_main = -1000
    pitch_main = 0
    yaw_main = 0
    roll_main = 0
    head = 0
    lost_focus_count = 0
    pitch_count = 0
    yaw_count = 0
    decay = 0.9 # use decay to smoothen the noise in feature values
    triggered = False


    count_EAR = 0
    count_MAR = 0

    max_yaw = 40
    min_yaw = -40

    lost_focus = False
    drowsy = False

    label = None
    processed_frames = 0
    missed_count = 0
    alert_release_at = 0.0
    drowsy_condition_started_at = None
    out_of_camera_announced = False

    input_data = []
    frame_before_run = 0
    count_decision = 0 
    cap, width, height = open_camera()
    if cap is None:
        state.mode = "idle"
        state.request_recalibrate = False
        return

    # cap.set(3, width)
    # cap.set(4, height)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    while state.running and not state.request_recalibrate:
        if not state.running_inference:
            state.mode = "infer_paused"
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None

            reset_inference_buffers()
            paused_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(paused_frame, "Inference paused", (int(0.15 * width), int(0.45 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(paused_frame, "Press D7 button to resume", (int(0.08 * width), int(0.58 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow('MediaPipe FaceMesh', paused_frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                state.request_recalibrate = True
            continue

        if cap is None or not cap.isOpened():
            cap, width, height = open_camera()
            if cap is None:
                time.sleep(0.5)
                continue

        state.mode = "infer"

        success, image = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        processed_frames += 1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image = run_face_mp(image, height=height, width=width)
        missed_count = missed_count + 1 if not detect else 0

        if state.running_inference: 
            
            if ear != -1000:
                ear = (ear - ears_norm[0])/ears_norm[1]
                mar = (mar - mars_norm[0])/mars_norm[1]
                puc = (puc - pucs_norm[0])/pucs_norm[1]
                moe = (moe - moes_norm[0])/moes_norm[1]
                pitch_main = pitch_pred - pitch_pred_norm
                yaw_main = yaw_pred - yaw_pred_norm
                roll_main = roll_pred - roll_pred_norm
                if ear_main == -1000:
                    ear_main = ear
                    mar_main = mar
                    puc_main = puc
                    moe_main = moe
                else:
                    ear_main = ear_main*decay + (1-decay)*ear #EMA data smoothing
                    mar_main = mar_main*decay + (1-decay)*mar
                    puc_main = puc_main*decay + (1-decay)*puc
                    moe_main = moe_main*decay + (1-decay)*moe
            else:
                ear_main = -1000
                mar_main = -1000    
                puc_main = -1000
                moe_main = -1000
                drowsy_condition_started_at = None


            if detect:
                out_of_camera_announced = False
                if pitch_main > 0.3 or pitch_main < - 0.2:
                    head = 1
                    pitch_count += 1
                else:
                    head = 0
                    pitch_count = 0
            else: 
                max_yaw = yaw_main if yaw_main > max_yaw else max_yaw
                min_yaw = yaw_main if yaw_main < min_yaw else min_yaw

                if pitch_main > 0.2 or pitch_main < -0.15:
                    pitch_count += 1
                    head = 1
                else:
                    pitch_count = 0

            if yaw_main > 0.2 or yaw_main < -0.2:
                yaw_count += 1
                # lost_focus_count += 1 if fast_movement
                lost_focus_count += 1
                lost_focus = True if lost_focus_count >= 40 else lost_focus
            else:
                lost_focus_count = 0
                lost_focus = False
                
            if len(input_data) == 20:
                input_data.pop(0)
            input_data.append([ear_main, mar_main, puc_main, moe_main])

            frame_before_run += 1
            
            if frame_before_run >= 15 and len(input_data) == 20:
                frame_before_run = 0
                label = get_classification(input_data) # 1 is drowsiness, 0 is normal
                if label == 0:
                    count_decision = 0 
                else:
                    count_decision += 1

            count_EAR = count_EAR + 1 if threshhold_EAR_MAR(ear_main, mar_main) and detect else 0

            cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "MAR: %.2f" %(mar_main), (int(0.27*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "MOE: %.2f" %(moe_main), (int(0.77*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Prepare text to display on the screen
            angle_text_pitch = f"Pitch: {pitch_main:.2f}"
            angle_text_yaw = f"Yaw: {yaw_main:.2f}"
            angle_text_roll = f"Roll: {roll_main:.2f}"

            # Display the angle values on the image
            cv2.putText(image, angle_text_pitch, (int(0.02*width), int(0.1*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, angle_text_yaw, (int(0.02*width), int(0.2*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, angle_text_roll, (int(0.02*width), int(0.3*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if label is not None:
                if label == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(image, "%s" %(states[label]), (int(0.4*image.shape[1]), int(0.22*image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)


            drowsy_detected = any(
                [
                    count_EAR > 30,
                    head == 1 and pitch_count >= 15,
                    count_decision >= state.count_alert_AI and detect,
                ]
            )

            if drowsy_detected:
                on_warning_drowsy()
            else:
                drowsy_condition_started_at = None
                if time.monotonic() >= alert_release_at:
                    off_warning_drowsy()

            if missed_count > 100:
                out_of_camera()
        else:
            image.fill(0)
            count_decision = 0
            pitch_count = 0
        cv2.imshow('MediaPipe FaceMesh', image)

        if processed_frames % 100 == 0:
            gc.collect()
            processed_frames = 0

        if cv2.waitKey(5) & 0xFF == ord("q"):
            state.request_recalibrate = True

    state.alert = False
    state.running_inference = True
    state.request_recalibrate = False
    state.mode = "idle"
    cv2.destroyAllWindows()
    if cap is not None and cap.isOpened():
        cap.release()
    # out.release()


if __name__ == "__main__":
    hardware_controller = None
    try:
        warnings.filterwarnings("ignore", category=UserWarning)

        right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
        left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
        mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
        states = ['normal', 'drowsy']

        hardware_controller = HardwareController()
        hardware_controller.start()

        # Khởi tạo FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.8,
            max_num_faces=1)
        mp_drawing = mp.solutions.drawing_utils 
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_head_pose = pickle.load(open(cfg.MODEL_HEAD_POSE, 'rb'))
        model_lstm_path = cfg.MODEL_FATIGUE
        model = torch.jit.load(model_lstm_path)
        model.eval()

        while True:
            if state.running:
                calibration_result = calibrate()
                if calibration_result is None:
                    time.sleep(0.1)
                    continue

                ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred, yaw_pred, roll_pred = calibration_result
                hardware_controller.announce_calibration_complete()
                hardware_controller.announce_infer_started()
                infer(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred, yaw_pred, roll_pred)
            else:
                state.mode = "idle"
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    except Exception as e:
        print("An error occurred:", e)
    
    finally:
        if hardware_controller is not None:
            hardware_controller.cleanup()
        cv2.destroyAllWindows()
