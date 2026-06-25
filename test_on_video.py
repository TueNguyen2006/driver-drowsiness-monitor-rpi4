import cv2
import math
import numpy as np
import torch
import pickle
import warnings
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
import state

warnings.filterwarnings("ignore", category=UserWarning)

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

MODEL_PATH = "/tmp/face_landmarker.task"
options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_tracking_confidence=0.8,
)
face_landmarker = FaceLandmarker.create_from_options(options)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_head_pose = pickle.load(open(cfg.MODEL_HEAD_POSE, 'rb'))
model = torch.jit.load(cfg.MODEL_FATIGUE)
model.eval()

right_eye_indices = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye_indices = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth_indices = [[61, 291], [39, 181], [0, 17], [269, 405]]
states_labels = ['normal', 'drowsy']

left_eye = left_eye_indices
right_eye = right_eye_indices
mouth = mouth_indices

detect = False

def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2

def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def pupil_circularity(landmarks, eye):
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
        distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
        distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
        distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
        distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
        distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
        distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
        distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + pupil_circularity(landmarks, right_eye)) / 2

def normalize_test(poses_array):
    normalized_array = poses_array.copy()
    for dim_idx in [0, 1]:
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = poses_array[feature_idx] - poses_array[dim_idx]
        diff = poses_array[12 + dim_idx] - poses_array[4 + dim_idx]
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = normalized_array[feature_idx] / diff
    return [normalized_array]

def head_pose(face_features):
    pitch_pred, yaw_pred, roll_pred = 0, 0, 0
    face_features_normalized = normalize_test(face_features)
    pitch_pred, yaw_pred, roll_pred = model_head_pose.predict(face_features_normalized).ravel()
    return pitch_pred, yaw_pred, roll_pred

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def run_face_mp(image):
    global detect
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291
    face_features = []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
        h, w = image.shape[:2]
        landmarks_list = results.face_landmarks[0]

        landmarks_positions = []
        for idx, lm in enumerate(landmarks_list):
            landmarks_positions.append([lm.x, lm.y])
            if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                face_features.append(lm.x)
                face_features.append(lm.y)

        landmarks_positions = np.array(landmarks_positions)
        pitch_pred, yaw_pred, roll_pred = head_pose(face_features)
        landmarks_positions[:, 0] *= w
        landmarks_positions[:, 1] *= h

        Nose_x = int(landmarks_positions[NOSE][0])
        Nose_y = int(landmarks_positions[NOSE][1])

        image = draw_axes(image, pitch_pred, yaw_pred, roll_pred, Nose_x, Nose_y)

        # Draw face mesh connections manually
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            start_idx, end_idx = connection.start, connection.end
            if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                pt1 = (int(landmarks_list[start_idx].x * w), int(landmarks_list[start_idx].y * h))
                pt2 = (int(landmarks_list[end_idx].x * w), int(landmarks_list[end_idx].y * h))
                cv2.line(image, pt1, pt2, (100, 200, 100), 1)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
        detect = True
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000
        pitch_pred, yaw_pred, roll_pred = 0, 0, 0
        detect = False
    return ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image

def calibrate_from_video(video_path, calib_frame_count=100):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0

    ears, mars, pucs, moes = [], [], [], []
    pitch_preds, yaw_preds, roll_preds = [], [], []

    while frames < calib_frame_count:
        success, image = cap.read()
        if not success:
            break
        frames += 1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, _ = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            pitch_preds.append(pitch_pred)
            yaw_preds.append(yaw_pred)
            roll_preds.append(roll_pred)

    cap.release()
    if not ears:
        return None

    return (
        [np.mean(ears), np.std(ears)],
        [np.mean(mars), np.std(mars)],
        [np.mean(pucs), np.std(pucs)],
        [np.mean(moes), np.std(moes)],
        np.mean(pitch_preds),
        np.mean(yaw_preds),
        np.mean(roll_preds),
    )

def get_classification(input_data):
    with torch.no_grad():
        model_input = [input_data[i:i + 5] for i in range(0, 10, 3)]
        model_input = torch.FloatTensor(np.array(model_input))
        preds = model(model_input)
        preds = (preds > 0.5).int().cpu().numpy()
    return int(preds.sum() >= 3)

def infer_on_video(video_path, ears_norm, mars_norm, pucs_norm, moes_norm,
                   pitch_pred_norm, yaw_pred_norm, roll_pred_norm,
                   output_video_path=None):
    global detect

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    ear_main = -1000
    mar_main = -1000
    puc_main = -1000
    moe_main = -1000
    pitch_main = 0
    yaw_main = 0
    roll_main = 0
    head = 0
    pitch_count = 0
    yaw_count = 0
    lost_focus_count = 0
    lost_focus = False
    decay = 0.9
    count_EAR = 0
    count_decision = 0
    frame_before_run = 0
    input_data = []
    label = None

    drowsy_frames = 0
    total_detected_frames = 0

    frame_idx = 0
    while True:
        success, image = cap.read()
        if not success:
            break
        frame_idx += 1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, img_out = run_face_mp(image)

        if detect:
            total_detected_frames += 1

        if ear != -1000:
            ear = (ear - ears_norm[0]) / ears_norm[1]
            mar = (mar - mars_norm[0]) / mars_norm[1]
            puc = (puc - pucs_norm[0]) / pucs_norm[1]
            moe = (moe - moes_norm[0]) / moes_norm[1]
            pitch_main = pitch_pred - pitch_pred_norm
            yaw_main = yaw_pred - yaw_pred_norm
            roll_main = roll_pred - roll_pred_norm
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main * decay + (1 - decay) * ear
                mar_main = mar_main * decay + (1 - decay) * mar
                puc_main = puc_main * decay + (1 - decay) * puc
                moe_main = moe_main * decay + (1 - decay) * moe

        if detect:
            if pitch_main > 0.3 or pitch_main < -0.2:
                head = 1
                pitch_count += 1
            else:
                head = 0
                pitch_count = 0

        if yaw_main > 0.2 or yaw_main < -0.2:
            yaw_count += 1
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
            label = get_classification(input_data)
            if label == 0:
                count_decision = 0
            else:
                count_decision += 1

        count_EAR = count_EAR + 1 if (ear_main < -2 or mar_main > 30) and detect else 0

        drowsy_detected = any([
            count_EAR > 30,
            head == 1 and pitch_count >= 15,
            count_decision >= state.count_alert_AI and detect,
        ])

        if drowsy_detected:
            drowsy_frames += 1

        cv2.putText(img_out, "EAR: %.2f" % ear_main, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img_out, "MAR: %.2f" % mar_main, (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img_out, "PUC: %.2f" % puc_main, (420, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if label is not None:
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.putText(img_out, "%s" % states_labels[label],
                        (int(0.4 * width), int(0.22 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        if drowsy_detected:
            cv2.putText(img_out, "DROWSY ALERT!",
                        (int(0.3 * width), int(0.9 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(img_out, "Frame: %d/%d" % (frame_idx, total_frames),
                    (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if out is not None:
            out.write(img_out)

        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100
            fps_r = frame_idx / (time.time() - start_time) if frame_idx > 1 else 0
            print(f"  Progress: {frame_idx}/{total_frames} ({pct:.0f}%) @ {fps_r:.1f} fps",
                  flush=True)

    cap.release()
    if out is not None:
        out.release()

    return {
        "total_frames": frame_idx,
        "drowsy_frames": drowsy_frames,
        "total_detected_frames": total_detected_frames,
        "drowsy_pct_detected": (drowsy_frames / total_detected_frames * 100) if total_detected_frames > 0 else 0,
        "face_detection_rate": (total_detected_frames / frame_idx * 100) if frame_idx > 0 else 0,
    }

def process_video(video_path, output_path=None):
    global start_time
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"{'='*60}")

    print("\n[1/3] Calibrating from first 100 frames...")
    calib_result = calibrate_from_video(video_path, calib_frame_count=100)
    if calib_result is None:
        print("ERROR: No face detected during calibration!")
        return None
    (ears_norm, mars_norm, pucs_norm, moes_norm,
     pitch_pred, yaw_pred, roll_pred) = calib_result
    print(f"  Calibration: EAR={ears_norm[0]:.3f}±{ears_norm[1]:.3f}, "
          f"MAR={mars_norm[0]:.3f}±{mars_norm[1]:.3f}, "
          f"PUC={pucs_norm[0]:.3f}±{pucs_norm[1]:.3f}")
    print(f"  Head pose norm: pitch={pitch_pred:.3f}, yaw={yaw_pred:.3f}, roll={roll_pred:.3f}")

    print("\n[2/3] Running inference on video...")
    stats = infer_on_video(video_path, ears_norm, mars_norm, pucs_norm, moes_norm,
                          pitch_pred, yaw_pred, roll_pred, output_video_path=output_path)

    elapsed = time.time() - start_time
    print(f"\n[3/3] Results (in {elapsed:.1f}s):")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Frames with face detected: {stats['total_detected_frames']} ({stats['face_detection_rate']:.1f}%)")
    print(f"  Frames flagged as drowsy: {stats['drowsy_frames']}")
    print(f"  Drowsy ratio (of detected frames): {stats['drowsy_pct_detected']:.1f}%")
    if output_path:
        print(f"  Output video saved: {output_path}")
    print()
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_on_video.py <video_path> [output_video_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    start_time = time.time()
    stats = process_video(video_path, output_path)

    if stats and stats['drowsy_pct_detected'] > 0:
        print(f"\n{'='*60}")
        print(f"CONCLUSION: System detected drowsiness in {stats['drowsy_frames']} frames "
              f"({stats['drowsy_pct_detected']:.1f}% of detected frames)")
        if stats['drowsy_pct_detected'] > 20:
            print("VERDICT: Video classified as DROWSY")
        else:
            print("VERDICT: Video classified as mostly ALERT (some drowsy moments)")
        print(f"{'='*60}")
    elif stats:
        print(f"\n{'='*60}")
        print("CONCLUSION: No drowsiness detected in this video")
        print("VERDICT: Video classified as ALERT")
        print(f"{'='*60}")
