import os
import json
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIG
# =========================================================
VIDEO_PATH = "/content/00002906.mp4"
OPENFACE_CSV_PATH = "/content/openface_out/00002906.csv"
POSE_MODEL_PATH = "/content/pose_landmarker_heavy.task"

OUTPUT_JSON_PATH = "/content/final_integrated_feature_summary.json"
PREVIEW_IMAGE_PATH = "/content/final_preview.jpg"
BLAZEPOSE_CSV_PATH = "/content/blazepose_landmarks.csv"

MIN_OPENFACE_CONFIDENCE = 0.8
SEGMENT_SECONDS = 2.0

SAMPLE_EVERY_N_FRAMES = 2
MIN_POSE_PRESENCE = 0.5


# =========================================================
# GENERIC HELPERS
# =========================================================
def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def round_or_none(x: Optional[float], digits: int = 4) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), digits)


def safe_mean(df: pd.DataFrame, cols: List[str]) -> float:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols or len(df) == 0:
        return 0.0
    return float(df[valid_cols].mean().mean())


def normalize_au_intensity(x: float, max_intensity: float = 5.0) -> float:
    return clamp(x / max_intensity)


def euclidean_2d(a, b) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


# =========================================================
# VIDEO / OPENCV
# =========================================================
def get_video_info(video_path: str) -> Dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_sec = 0.0
    if fps and fps > 0 and frame_count > 0:
        duration_sec = frame_count / fps

    cap.release()

    return {
        "video_path": video_path,
        "file_name": os.path.basename(video_path),
        "fps": round_or_none(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round_or_none(duration_sec, 3),
    }


def save_preview_frame(video_path: str, out_path: str, timestamp_sec: Optional[float] = None) -> Optional[str]:
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    if timestamp_sec is None:
        duration_sec = frame_count / fps if frame_count > 0 else 0.0
        timestamp_sec = duration_sec * 0.5 if duration_sec > 0 else 0.0

    target_frame = int(timestamp_sec * fps)
    target_frame = max(0, min(target_frame, max(0, frame_count - 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame)
    return out_path


# =========================================================
# OPENFACE LOADING / FILTERING
# =========================================================
def load_openface_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def filter_openface_df(df: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    filtered = df.copy()

    if "success" in filtered.columns:
        filtered = filtered[filtered["success"] == 1].copy()

    if "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= min_confidence].copy()

    if len(filtered) == 0:
        raise ValueError("No valid OpenFace rows after filtering.")

    return filtered


# =========================================================
# OPENFACE FEATURE SUMMARY
# =========================================================
def compute_openface_tracking_features(filtered_df: pd.DataFrame, raw_total_rows: int) -> Dict:
    valid_frames = len(filtered_df)
    face_tracking_pct = (valid_frames / raw_total_rows * 100.0) if raw_total_rows > 0 else None
    mean_confidence = float(filtered_df["confidence"].mean()) if "confidence" in filtered_df.columns else None

    return {
        "raw_csv_rows": int(raw_total_rows),
        "valid_frames": int(valid_frames),
        "face_tracking_pct": round_or_none(face_tracking_pct, 2),
        "mean_confidence": round_or_none(mean_confidence, 4),
    }


def compute_openface_attention_features(df: pd.DataFrame) -> Dict:
    if not {"gaze_angle_x", "gaze_angle_y"}.issubset(df.columns) or len(df) == 0:
        return {
            "gaze_angle_x_mean_abs": None,
            "gaze_angle_y_mean_abs": None,
            "gaze_angle_x_std": None,
            "gaze_angle_y_std": None,
            "forward_attention_pct": None,
        }

    gaze_x_mean_abs = float(df["gaze_angle_x"].abs().mean())
    gaze_y_mean_abs = float(df["gaze_angle_y"].abs().mean())
    gaze_x_std = float(df["gaze_angle_x"].std())
    gaze_y_std = float(df["gaze_angle_y"].std())

    forward = df[
        (df["gaze_angle_x"].abs() < 0.25) &
        (df["gaze_angle_y"].abs() < 0.25)
    ]
    forward_attention_pct = (len(forward) / len(df)) * 100.0 if len(df) > 0 else None

    return {
        "gaze_angle_x_mean_abs": round_or_none(gaze_x_mean_abs, 4),
        "gaze_angle_y_mean_abs": round_or_none(gaze_y_mean_abs, 4),
        "gaze_angle_x_std": round_or_none(gaze_x_std, 4),
        "gaze_angle_y_std": round_or_none(gaze_y_std, 4),
        "forward_attention_pct": round_or_none(forward_attention_pct, 2),
    }


def compute_openface_head_pose_features(df: pd.DataFrame) -> Dict:
    pose_cols = [c for c in ["pose_Rx", "pose_Ry", "pose_Rz"] if c in df.columns]

    if not pose_cols or len(df) == 0:
        return {
            "pose_Rx_mean_abs": None,
            "pose_Ry_mean_abs": None,
            "pose_Rz_mean_abs": None,
            "pose_Rx_std": None,
            "pose_Ry_std": None,
            "pose_Rz_std": None,
            "head_stability_pct": None,
        }

    pose_rx_mean_abs = float(df["pose_Rx"].abs().mean()) if "pose_Rx" in df.columns else None
    pose_ry_mean_abs = float(df["pose_Ry"].abs().mean()) if "pose_Ry" in df.columns else None
    pose_rz_mean_abs = float(df["pose_Rz"].abs().mean()) if "pose_Rz" in df.columns else None

    pose_rx_std = float(df["pose_Rx"].std()) if "pose_Rx" in df.columns else None
    pose_ry_std = float(df["pose_Ry"].std()) if "pose_Ry" in df.columns else None
    pose_rz_std = float(df["pose_Rz"].std()) if "pose_Rz" in df.columns else None

    rot_std = float(df[pose_cols].std().mean())
    head_stability_pct = max(0.0, min(100.0, 100.0 * (1.0 - rot_std / 0.30)))

    return {
        "pose_Rx_mean_abs": round_or_none(pose_rx_mean_abs, 4),
        "pose_Ry_mean_abs": round_or_none(pose_ry_mean_abs, 4),
        "pose_Rz_mean_abs": round_or_none(pose_rz_mean_abs, 4),
        "pose_Rx_std": round_or_none(pose_rx_std, 4),
        "pose_Ry_std": round_or_none(pose_ry_std, 4),
        "pose_Rz_std": round_or_none(pose_rz_std, 4),
        "head_stability_pct": round_or_none(head_stability_pct, 2),
    }


def compute_openface_au_features(df: pd.DataFrame, top_n: int = 10) -> Dict:
    au_r_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    au_c_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]

    if len(df) == 0 or not au_r_cols:
        return {
            "avg_au_intensity": None,
            "avg_au_variance": None,
            "facial_activity_pct": None,
            "top_action_units_by_mean_intensity": {},
            "au_presence_rates": {},
        }

    avg_au_intensity = float(df[au_r_cols].mean().mean())
    avg_au_variance = float(df[au_r_cols].var().mean())
    facial_activity_pct = max(0.0, min(100.0, 100.0 * min(avg_au_variance / 2.0, 1.0)))

    top_aus = df[au_r_cols].mean().sort_values(ascending=False).head(top_n)
    top_action_units = {k: round(float(v), 3) for k, v in top_aus.items()}

    au_presence_rates = {}
    if au_c_cols:
        for col in au_c_cols:
            au_presence_rates[col] = round(float(df[col].mean()), 4)

    return {
        "avg_au_intensity": round_or_none(avg_au_intensity, 4),
        "avg_au_variance": round_or_none(avg_au_variance, 4),
        "facial_activity_pct": round_or_none(facial_activity_pct, 2),
        "top_action_units_by_mean_intensity": top_action_units,
        "au_presence_rates": au_presence_rates,
    }


# =========================================================
# OPENFACE EMOTION TENDENCY FEATURES
# =========================================================
def score_happiness(df: pd.DataFrame) -> float:
    au06 = normalize_au_intensity(safe_mean(df, ["AU06_r"]))
    au12 = normalize_au_intensity(safe_mean(df, ["AU12_r"]))
    au25 = normalize_au_intensity(safe_mean(df, ["AU25_r"]))
    return clamp(0.45 * au06 + 0.45 * au12 + 0.10 * au25)


def score_sadness(df: pd.DataFrame) -> float:
    au01 = normalize_au_intensity(safe_mean(df, ["AU01_r"]))
    au04 = normalize_au_intensity(safe_mean(df, ["AU04_r"]))
    au15 = normalize_au_intensity(safe_mean(df, ["AU15_r"]))
    return clamp(0.35 * au01 + 0.35 * au04 + 0.30 * au15)


def score_surprise(df: pd.DataFrame) -> float:
    au01 = normalize_au_intensity(safe_mean(df, ["AU01_r"]))
    au02 = normalize_au_intensity(safe_mean(df, ["AU02_r"]))
    au05 = normalize_au_intensity(safe_mean(df, ["AU05_r"]))
    au26 = normalize_au_intensity(safe_mean(df, ["AU26_r"]))
    return clamp(0.20 * au01 + 0.20 * au02 + 0.30 * au05 + 0.30 * au26)


def score_anger_tension(df: pd.DataFrame) -> float:
    au04 = normalize_au_intensity(safe_mean(df, ["AU04_r"]))
    au05 = normalize_au_intensity(safe_mean(df, ["AU05_r"]))
    au07 = normalize_au_intensity(safe_mean(df, ["AU07_r"]))
    au23 = normalize_au_intensity(safe_mean(df, ["AU23_r"]))
    return clamp(0.30 * au04 + 0.20 * au05 + 0.25 * au07 + 0.25 * au23)


def score_disgust(df: pd.DataFrame) -> float:
    au09 = normalize_au_intensity(safe_mean(df, ["AU09_r"]))
    au10 = normalize_au_intensity(safe_mean(df, ["AU10_r"]))
    return clamp(0.50 * au09 + 0.50 * au10)


def score_fear_stress(df: pd.DataFrame) -> float:
    au01 = normalize_au_intensity(safe_mean(df, ["AU01_r"]))
    au02 = normalize_au_intensity(safe_mean(df, ["AU02_r"]))
    au04 = normalize_au_intensity(safe_mean(df, ["AU04_r"]))
    au05 = normalize_au_intensity(safe_mean(df, ["AU05_r"]))
    au20 = normalize_au_intensity(safe_mean(df, ["AU20_r"]))
    au26 = normalize_au_intensity(safe_mean(df, ["AU26_r"]))
    return clamp(
        0.15 * au01 +
        0.15 * au02 +
        0.20 * au04 +
        0.20 * au05 +
        0.15 * au20 +
        0.15 * au26
    )


def score_neutral(df: pd.DataFrame) -> float:
    au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    if not au_cols or len(df) == 0:
        return 0.0
    avg_au = float(df[au_cols].mean().mean())
    return clamp(1.0 - normalize_au_intensity(avg_au))


def compute_openface_emotion_tendency_features(df: pd.DataFrame) -> Dict:
    return {
        "happiness": round(float(score_happiness(df)), 4),
        "sadness": round(float(score_sadness(df)), 4),
        "surprise": round(float(score_surprise(df)), 4),
        "anger_tension": round(float(score_anger_tension(df)), 4),
        "disgust": round(float(score_disgust(df)), 4),
        "fear_stress": round(float(score_fear_stress(df)), 4),
        "neutral": round(float(score_neutral(df)), 4),
    }


def compute_openface_timeline(df: pd.DataFrame, segment_seconds: float) -> List[Dict]:
    if "timestamp" not in df.columns or len(df) == 0:
        return []

    start_t = float(df["timestamp"].min())
    end_t = float(df["timestamp"].max())
    segments = []
    current = start_t

    while current < end_t:
        nxt = current + segment_seconds
        seg = df[(df["timestamp"] >= current) & (df["timestamp"] < nxt)].copy()

        if len(seg) > 0:
            segments.append({
                "start_sec": round(current, 2),
                "end_sec": round(nxt, 2),
                "num_frames": int(len(seg)),
                "attention": compute_openface_attention_features(seg),
                "head_pose": compute_openface_head_pose_features(seg),
                "action_units": compute_openface_au_features(seg, top_n=5),
                "emotion_tendencies": compute_openface_emotion_tendency_features(seg),
            })

        current = nxt

    return segments


# =========================================================
# BLAZEPOSE / MEDIAPIPE
# =========================================================
def create_pose_landmarker(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_pose_rows(video_path: str, model_path: str, sample_every_n_frames: int = 2) -> List[Dict]:
    mp_image = mp.Image
    mp_image_format = mp.ImageFormat

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    rows: List[Dict] = []
    frame_idx = 0

    with create_pose_landmarker(model_path) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_every_n_frames != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp_image(image_format=mp_image_format.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_frame, timestamp_ms)

            row = {
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 4),
                "pose_detected": 0,
            }

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose = result.pose_landmarks[0]
                world_pose = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                row["pose_detected"] = 1

                for i, lm in enumerate(pose):
                    row[f"lm_{i}_x"] = float(lm.x)
                    row[f"lm_{i}_y"] = float(lm.y)
                    row[f"lm_{i}_z"] = float(lm.z)
                    row[f"lm_{i}_visibility"] = float(getattr(lm, "visibility", 0.0))
                    row[f"lm_{i}_presence"] = float(getattr(lm, "presence", 0.0))

                if world_pose:
                    for i, lm in enumerate(world_pose):
                        row[f"world_{i}_x"] = float(lm.x)
                        row[f"world_{i}_y"] = float(lm.y)
                        row[f"world_{i}_z"] = float(lm.z)
                        row[f"world_{i}_visibility"] = float(getattr(lm, "visibility", 0.0))
                        row[f"world_{i}_presence"] = float(getattr(lm, "presence", 0.0))

            rows.append(row)
            frame_idx += 1

    cap.release()
    return rows


def landmark_present(row: pd.Series, idx: int, threshold: float = 0.5) -> bool:
    vis = row.get(f"lm_{idx}_visibility", 0.0)
    pres = row.get(f"lm_{idx}_presence", 0.0)
    return (vis >= threshold) or (pres >= threshold)


def get_xy(row: pd.Series, idx: int):
    return (row[f"lm_{idx}_x"], row[f"lm_{idx}_y"])


def compute_pose_tracking_metrics(df: pd.DataFrame) -> Dict:
    total = len(df)
    detected = int(df["pose_detected"].sum()) if "pose_detected" in df.columns else 0
    pose_tracking_pct = safe_div(detected, total) * 100.0 if total > 0 else 0.0
    return {
        "processed_frames": total,
        "pose_detected_frames": detected,
        "pose_tracking_pct": round_or_none(pose_tracking_pct, 2),
    }


def compute_shoulder_symmetry(df: pd.DataFrame) -> Dict:
    diffs = []
    for _, row in df.iterrows():
        if landmark_present(row, 11, MIN_POSE_PRESENCE) and landmark_present(row, 12, MIN_POSE_PRESENCE):
            ly = row["lm_11_y"]
            ry = row["lm_12_y"]
            diffs.append(abs(ly - ry))

    if not diffs:
        return {"mean_shoulder_y_diff": None, "shoulder_symmetry_score_pct": None}

    mean_diff = float(np.mean(diffs))
    score = max(0.0, min(100.0, 100.0 * (1.0 - mean_diff / 0.08)))
    return {
        "mean_shoulder_y_diff": round_or_none(mean_diff, 4),
        "shoulder_symmetry_score_pct": round_or_none(score, 2),
    }


def compute_head_torso_alignment(df: pd.DataFrame) -> Dict:
    offsets = []
    for _, row in df.iterrows():
        needed = [0, 11, 12]
        if all(landmark_present(row, i, MIN_POSE_PRESENCE) for i in needed):
            nose_x = row["lm_0_x"]
            shoulder_mid_x = (row["lm_11_x"] + row["lm_12_x"]) / 2.0
            offsets.append(abs(nose_x - shoulder_mid_x))

    if not offsets:
        return {"mean_head_torso_x_offset": None, "head_alignment_score_pct": None}

    mean_offset = float(np.mean(offsets))
    score = max(0.0, min(100.0, 100.0 * (1.0 - mean_offset / 0.15)))
    return {
        "mean_head_torso_x_offset": round_or_none(mean_offset, 4),
        "head_alignment_score_pct": round_or_none(score, 2),
    }


def compute_torso_uprightness(df: pd.DataFrame) -> Dict:
    lean_vals = []
    for _, row in df.iterrows():
        needed = [11, 12, 23, 24]
        if all(landmark_present(row, i, MIN_POSE_PRESENCE) for i in needed):
            shoulder_mid_x = (row["lm_11_x"] + row["lm_12_x"]) / 2.0
            shoulder_mid_y = (row["lm_11_y"] + row["lm_12_y"]) / 2.0
            hip_mid_x = (row["lm_23_x"] + row["lm_24_x"]) / 2.0
            hip_mid_y = (row["lm_23_y"] + row["lm_24_y"]) / 2.0

            dx = abs(shoulder_mid_x - hip_mid_x)
            dy = abs(shoulder_mid_y - hip_mid_y)
            if dy > 1e-6:
                lean_vals.append(dx / dy)

    if not lean_vals:
        return {"mean_torso_lean_ratio": None, "torso_upright_score_pct": None}

    mean_lean = float(np.mean(lean_vals))
    score = max(0.0, min(100.0, 100.0 * (1.0 - mean_lean / 0.35)))
    return {
        "mean_torso_lean_ratio": round_or_none(mean_lean, 4),
        "torso_upright_score_pct": round_or_none(score, 2),
    }


def compute_hand_visibility(df: pd.DataFrame) -> Dict:
    left_visible = 0
    right_visible = 0
    total = len(df)

    for _, row in df.iterrows():
        if landmark_present(row, 15, MIN_POSE_PRESENCE):
            left_visible += 1
        if landmark_present(row, 16, MIN_POSE_PRESENCE):
            right_visible += 1

    return {
        "left_wrist_visibility_pct": round_or_none(safe_div(left_visible, total) * 100.0, 2),
        "right_wrist_visibility_pct": round_or_none(safe_div(right_visible, total) * 100.0, 2),
        "any_hand_visibility_pct": round_or_none(
            safe_div(sum(
                1 for _, row in df.iterrows()
                if landmark_present(row, 15, MIN_POSE_PRESENCE) or landmark_present(row, 16, MIN_POSE_PRESENCE)
            ), total) * 100.0, 2
        ),
    }


def compute_movement_stability(df: pd.DataFrame) -> Dict:
    shoulder_moves = []
    wrist_moves = []

    prev_shoulder_mid = None
    prev_left_wrist = None
    prev_right_wrist = None

    for _, row in df.iterrows():
        shoulder_mid = None
        if landmark_present(row, 11, MIN_POSE_PRESENCE) and landmark_present(row, 12, MIN_POSE_PRESENCE):
            shoulder_mid = (
                (row["lm_11_x"] + row["lm_12_x"]) / 2.0,
                (row["lm_11_y"] + row["lm_12_y"]) / 2.0,
            )

        if shoulder_mid is not None and prev_shoulder_mid is not None:
            shoulder_moves.append(euclidean_2d(prev_shoulder_mid, shoulder_mid))
        prev_shoulder_mid = shoulder_mid

        if landmark_present(row, 15, MIN_POSE_PRESENCE):
            cur_left = get_xy(row, 15)
            if prev_left_wrist is not None:
                wrist_moves.append(euclidean_2d(prev_left_wrist, cur_left))
            prev_left_wrist = cur_left
        else:
            prev_left_wrist = None

        if landmark_present(row, 16, MIN_POSE_PRESENCE):
            cur_right = get_xy(row, 16)
            if prev_right_wrist is not None:
                wrist_moves.append(euclidean_2d(prev_right_wrist, cur_right))
            prev_right_wrist = cur_right
        else:
            prev_right_wrist = None

    shoulder_move_mean = float(np.mean(shoulder_moves)) if shoulder_moves else None
    wrist_move_mean = float(np.mean(wrist_moves)) if wrist_moves else None

    body_stability_score = None
    if shoulder_move_mean is not None:
        body_stability_score = max(0.0, min(100.0, 100.0 * (1.0 - shoulder_move_mean / 0.03)))

    return {
        "mean_shoulder_midpoint_motion": round_or_none(shoulder_move_mean, 4),
        "mean_wrist_motion": round_or_none(wrist_move_mean, 4),
        "body_stability_score_pct": round_or_none(body_stability_score, 2),
    }


def compute_blazepose_feature_summary(video_path: str, model_path: str) -> Dict:
    rows = extract_pose_rows(
        video_path=video_path,
        model_path=model_path,
        sample_every_n_frames=SAMPLE_EVERY_N_FRAMES,
    )

    if not rows:
        raise RuntimeError("No BlazePose rows were extracted.")

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(BLAZEPOSE_CSV_PATH), exist_ok=True)
    df.to_csv(BLAZEPOSE_CSV_PATH, index=False)

    metrics = {
        "tracking": compute_pose_tracking_metrics(df),
        "shoulder_symmetry": compute_shoulder_symmetry(df),
        "head_alignment": compute_head_torso_alignment(df),
        "torso_uprightness": compute_torso_uprightness(df),
        "hand_visibility": compute_hand_visibility(df),
        "movement_stability": compute_movement_stability(df),
    }

    vals = [
        metrics["shoulder_symmetry"].get("shoulder_symmetry_score_pct"),
        metrics["head_alignment"].get("head_alignment_score_pct"),
        metrics["torso_uprightness"].get("torso_upright_score_pct"),
        metrics["movement_stability"].get("body_stability_score_pct"),
    ]
    vals = [v for v in vals if v is not None]
    overall_body_pose_score_pct = round(float(np.mean(vals)), 2) if vals else None

    return {
        "model_path": model_path,
        "sample_every_n_frames": SAMPLE_EVERY_N_FRAMES,
        "landmarks_csv_path": BLAZEPOSE_CSV_PATH,
        "metrics": metrics,
        "overall_body_pose_score_pct": overall_body_pose_score_pct,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not os.path.exists(OPENFACE_CSV_PATH):
        raise FileNotFoundError(f"OpenFace CSV not found: {OPENFACE_CSV_PATH}")
    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(f"Pose model not found: {POSE_MODEL_PATH}")

    video_info = get_video_info(VIDEO_PATH)
    preview_image_path = save_preview_frame(VIDEO_PATH, PREVIEW_IMAGE_PATH)

    raw_openface_df = load_openface_csv(OPENFACE_CSV_PATH)
    filtered_openface_df = filter_openface_df(raw_openface_df, MIN_OPENFACE_CONFIDENCE)

    openface_summary = {
        "input": {
            "csv_path": OPENFACE_CSV_PATH,
            "raw_csv_rows": int(len(raw_openface_df)),
            "min_confidence_filter": MIN_OPENFACE_CONFIDENCE,
            "segment_seconds": SEGMENT_SECONDS,
        },
        "tracking": compute_openface_tracking_features(filtered_openface_df, raw_total_rows=len(raw_openface_df)),
        "attention": compute_openface_attention_features(filtered_openface_df),
        "head_pose": compute_openface_head_pose_features(filtered_openface_df),
        "action_units": compute_openface_au_features(filtered_openface_df, top_n=10),
        "emotion_tendencies": compute_openface_emotion_tendency_features(filtered_openface_df),
        "timeline": compute_openface_timeline(filtered_openface_df, segment_seconds=SEGMENT_SECONDS),
    }

    blazepose_summary = compute_blazepose_feature_summary(VIDEO_PATH, POSE_MODEL_PATH)

    result = {
        "video_info": video_info,
        "preview_image_path": preview_image_path,
        "openface_summary": openface_summary,
        "blazepose_summary": blazepose_summary,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved integrated feature summary to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
