import os
from pathlib import Path

import cv2
import mediapipe as mp
import math
import numpy as np
import json

import warnings
import config as cfg
import pickle
import joblib

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings("ignore", category=UserWarning)
# --- CONFIGURATION & CONSTANTS ---
RIGHT_EYE_IDX = [[33, 133], [160, 144], [159, 145], [158, 153]]
LEFT_EYE_IDX = [[263, 362], [387, 373], [386, 374], [385, 380]]
MOUTH_IDX = [[61, 291], [39, 181], [0, 17], [269, 405]]

MP_FACE_MESH = mp.solutions.face_mesh
face_mesh = MP_FACE_MESH.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.8
)

# --- UTILITY FUNCTIONS ---

def calculate_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)."""
    return np.linalg.norm(p1[:2] - p2[:2])

def get_aspect_ratio(landmarks, points_idx):
    """Tính tỉ lệ (N1+N2+N3) / (3*D) cho mắt hoặc miệng."""
    n1 = calculate_distance(landmarks[points_idx[1][0]], landmarks[points_idx[1][1]])
    n2 = calculate_distance(landmarks[points_idx[2][0]], landmarks[points_idx[2][1]])
    n3 = calculate_distance(landmarks[points_idx[3][0]], landmarks[points_idx[3][1]])
    d = calculate_distance(landmarks[points_idx[0][0]], landmarks[points_idx[0][1]])
    return (n1 + n2 + n3) / (3 * d)

def get_pupil_circularity(landmarks, eye_idx):
    """Tính độ tròn của con ngươi."""
    # Tính chu vi dựa trên các điểm quanh mắt
    perimeter = (
        calculate_distance(landmarks[eye_idx[0][0]], landmarks[eye_idx[1][0]]) +
        calculate_distance(landmarks[eye_idx[1][0]], landmarks[eye_idx[2][0]]) +
        calculate_distance(landmarks[eye_idx[2][0]], landmarks[eye_idx[3][0]]) +
        calculate_distance(landmarks[eye_idx[3][0]], landmarks[eye_idx[0][1]]) +
        calculate_distance(landmarks[eye_idx[0][1]], landmarks[eye_idx[3][1]]) +
        calculate_distance(landmarks[eye_idx[3][1]], landmarks[eye_idx[2][1]]) +
        calculate_distance(landmarks[eye_idx[2][1]], landmarks[eye_idx[1][1]]) +
        calculate_distance(landmarks[eye_idx[1][1]], landmarks[eye_idx[0][0]])
    )
    area = math.pi * ((calculate_distance(landmarks[eye_idx[1][0]], landmarks[eye_idx[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

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
    face_features_normalized = normalize_test(face_features)
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

def run_face_mp(image):
    """Xử lý ảnh qua MediaPipe và trả về các đặc trưng mệt mỏi cùng Head Pose."""
    height, width = image.shape[:2]
    img_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Định nghĩa các chỉ số landmark
    IDXS = {
        'NOSE': 1, 'FOREHEAD': 10, 'LEFT_EYE': 33, 
        'MOUTH_LEFT': 61, 'CHIN': 199, 'RIGHT_EYE': 263, 'MOUTH_RIGHT': 291
    }

    results = face_mesh.process(img_rgb)
    
    # Giá trị mặc định khi không thấy mặt
    ear, mar, puc, moe = -1000, -1000, -1000, -1000
    p_pred, y_pred, r_pred = 0, 0, 0

    if results.multi_face_landmarks:
        # 1. Lấy tọa độ chuẩn hóa (0-1) để tính Head Pose và Feature Engineering
        # landmarks_norm dùng để tính head_pose (chưa nhân width/height)
        landmarks_norm = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
        
        # Tạo list face_features gồm x1, y1, x2, y2... từ 7 điểm quan trọng
        face_features = []
        for key in ['FOREHEAD', 'NOSE', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'CHIN', 'LEFT_EYE', 'RIGHT_EYE']:
            idx = IDXS[key]
            face_features.extend([landmarks_norm[idx][0], landmarks_norm[idx][1]])
        
        # --- TÍNH HEAD POSE ---
        # Giả sử hàm head_pose của bạn nhận vào list 14 phần tử (7 điểm * 2)
        p_pred, y_pred, r_pred = head_pose(face_features)

        # 2. Chuyển sang tọa độ Pixel để vẽ và tính EAR, MAR, PUC
        mesh_coords = landmarks_norm.copy()
        mesh_coords[:, 0] *= width
        mesh_coords[:, 1] *= height
        
        # --- TÍNH TOÁN ĐẶC TRƯNG ---
        # Sử dụng mesh_coords (pixel) cho các hàm cũ của bạn
        ear_l = get_aspect_ratio(mesh_coords, LEFT_EYE_IDX)
        ear_r = get_aspect_ratio(mesh_coords, RIGHT_EYE_IDX)
        ear = (ear_l + ear_r) / 2
        
        # MOUTH_IDX bạn cần định nghĩa cụ thể, ở đây giả sử dùng trung bình miệng
        mar = get_aspect_ratio(mesh_coords, MOUTH_IDX) # Ví dụ index 13 là tâm môi 
        
        puc_l = get_pupil_circularity(mesh_coords, LEFT_EYE_IDX)
        puc_r = get_pupil_circularity(mesh_coords, RIGHT_EYE_IDX)
        puc = (puc_l + puc_r) / 2
        
        moe = mar / ear if ear > 0 else 0


    return ear, mar, puc, moe, p_pred, y_pred, r_pred, image

# --- CORE PROCESSES ---

def calibrate(file_path, calib_frame_count=200):
    """Thực hiện cân chỉnh để lấy giá trị trung bình và độ lệch chuẩn."""
    features = {'ear': [], 'mar': [], 'puc': [], 'moe': []}
    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)

    while len(features['ear']) < calib_frame_count:
        success, image = cap.read()
        if not success: break
        
        ear, mar, puc, moe, yaw, pitch, roll, _ = run_face_mp(image)
        if ear != -1000:
            features['ear'].append(ear)
            features['mar'].append(mar)
            features['puc'].append(puc)
            features['moe'].append(moe)
            
        if cv2.waitKey(5) & 0xFF == ord("q"): break

    cap.release()
    
    # Trả về [mean, std] cho từng đặc trưng
    results = []
    for key in ['ear', 'mar', 'puc', 'moe']:
        data = np.array(features[key])
        results.append([data.mean(), data.std()])
    return results

def infer(video_files, frames_take_for_dataset=8000, filter_label=False, output_name="dataset_final.json"):
    """
    Thực hiện inference và lưu kết quả vào file JSON.
    :param filter_label: Nếu True, áp dụng logic tính toán lại label dựa trên EAR. 
                         Nếu False, dùng label mặc định từ video (0 hoặc 1).
    """
    decay = 0.9
    file_dataset = []
    
    for video_pair in video_files:
        # Bước 1: Calibrate với video đầu tiên (thường là video tỉnh táo - label 0)
        print(f"Calibrating with: {video_pair[0]}")
        norms = calibrate(video_pair[0])
        print(f"norms (mean, std) for ear, mar, puc, moe: {norms}")
        ear_n, mar_n, puc_n, moe_n = norms

        # Bước 2: Xử lý từng video trong cặp (video 0 và video 1)
        for label_idx in [0, 1]:
            video_name = video_pair[label_idx]
            cap = cv2.VideoCapture(video_name)
            cap.set(cv2.CAP_PROP_POS_MSEC, 20000) # Bỏ qua 20s đầu
            
            input_data_window = []
            ear_m, mar_m, puc_m, moe_m = [-1000] * 4
            yaw_m, pitch_m, roll_m = [-1000] * 3
            frame_count = 0
            stride_counter = 0
            sample_id = 0

            print(f"Processing: {video_name} (Default Label: {label_idx})")

            while frame_count < frames_take_for_dataset:
                success, image = cap.read()
                if not success: break
                
                frame_count += 1
                ear, mar, puc, moe, yaw, pitch, roll, _ = run_face_mp(image)

                if ear != -1000:
                    # 1. Chuẩn hóa các đặc trưng mắt/miệng (dựa trên calibration)
                    ear = (ear - ear_n[0]) / ear_n[1]
                    mar = (mar - mar_n[0]) / mar_n[1]
                    puc = (puc - puc_n[0]) / puc_n[1]
                    moe = (moe - moe_n[0]) / moe_n[1]
                    
                    # 2. Làm mượt bằng EMA (Áp dụng cho cả 7 đặc trưng)
                    if ear_m == -1000:
                        # Khởi tạo frame đầu tiên
                        ear_m, mar_m, puc_m, moe_m = ear, mar, puc, moe
                        yaw_m, pitch_m, roll_m = yaw, pitch, roll
                    else:
                        # Cập nhật giá trị mượt cho diện mạo
                        ear_m = ear_m * decay + (1 - decay) * ear
                        mar_m = mar_m * decay + (1 - decay) * mar
                        puc_m = puc_m * decay + (1 - decay) * puc
                        moe_m = moe_m * decay + (1 - decay) * moe

                else:
                    # Nếu không detect được mặt, reset toàn bộ về giá trị lỗi
                    ear_m, mar_m, puc_m, moe_m = [-1000] * 4
                    yaw_m, pitch_m, roll_m = [-1000] * 3

                    cv2.imshow('MediaPipe FaceMesh', image)

                if cv2.waitKey(1) & 0xFF == ord("q"): break
            
            cap.release()


# --- EXECUTION ---
if __name__ == "__main__":
    model_head_pose = pickle.load(open(cfg.MODEL_HEAD_POSE, 'rb'))
    # model_fatigue = joblib.load(cfg.MODEL_FATIGUE)
    dataset_dir = Path(os.getenv("DROWSINESS_DATASET_DIR", "dataset"))
    videos = [
        [
            str(dataset_dir / "Dash" / "Female" / "1-FemaleNoGlasses.avi"),
            str(dataset_dir / "Dash" / "Female" / "2-FemaleNoGlasses.avi"),
        ],
        # ... Thêm các file khác vào đây
    ]
    
    infer(videos, frames_take_for_dataset=5000)
