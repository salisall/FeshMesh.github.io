#!/usr/bin/env python3
"""
Live webcam: MediaPipe face mesh + hands.

- Face mesh: full tessellation, contour edges, iris rings (styled drawers — teal mesh, coloured iris).
- Hands: up to 2 hands, 21 landmarks with connections (amber points, white connections).
- Posed expression guess (landmark rules): happy / sad / surprised / angry / disgust / neutral.
- Mirrored feed; FPS + expression HUD (top-left). C = set neutral baseline, Q = quit.

refine_landmarks=True keeps iris/lip refinement (468 → 478 landmarks). Set False for a lighter model.
Change cv2.VideoCapture(0) if you need another camera index.

Outputs are rough heuristics for exaggerated poses, not reliable emotion inference.

Accuracy tips: press C with a neutral face (camera steady). Works best with clear,
frontal lighting and exaggerated poses.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# BGR — amber-ish landmarks, white bone connections
HAND_LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=(0, 165, 255),
    thickness=2,
    circle_radius=4,
)
HAND_CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=2,
    circle_radius=2,
)

# MediaPipe FaceMesh indices (468 topology; unchanged with refine_landmarks).
_LIP_U, _LIP_L = 13, 14
_MOUTH_L, _MOUTH_R = 61, 291
_NOSE_TIP = 1
_EYE_INNER_L, _EYE_INNER_R = 362, 133
_EYE_OUT_L, _EYE_OUT_R = 263, 33
_EYE_TOP_L, _EYE_BOT_L = 386, 374
_EYE_TOP_R, _EYE_BOT_R = 159, 145
_BROW_L, _BROW_R = 285, 55
_BROW_INNER_L, _BROW_INNER_R = 336, 107
_CHIN, _FOREHEAD = 152, 10


def _pt(lm, idx: int) -> Tuple[float, float]:
    p = lm.landmark[idx]
    return p.x, p.y


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_posed_expression_features(lm) -> Optional[Dict[str, float]]:
    """Geometry: inter-eye scale for most metrics; smile arch uses face height (pitch-robust)."""
    scale = _dist(_pt(lm, _EYE_INNER_L), _pt(lm, _EYE_INNER_R))
    if scale < 1e-5:
        return None

    face_h = _dist(_pt(lm, _CHIN), _pt(lm, _FOREHEAD))
    if face_h < 1e-5:
        face_h = scale

    nose = _pt(lm, _NOSE_TIP)
    ml, mr = _pt(lm, _MOUTH_L), _pt(lm, _MOUTH_R)
    lip_u, lip_l = _pt(lm, _LIP_U), _pt(lm, _LIP_L)
    corners_mid = ((ml[0] + mr[0]) * 0.5, (ml[1] + mr[1]) * 0.5)
    lip_mid_y = (lip_u[1] + lip_l[1]) * 0.5

    mouth_open = _dist(lip_u, lip_l) / scale
    mouth_width = _dist(ml, mr) / scale
    # Corners move up when smiling; compare to lip midline (less head-pitch sensitive than nose).
    smile_arch = (lip_mid_y - corners_mid[1]) / max(face_h, 1e-6)

    brow_l = _pt(lm, _BROW_L)
    brow_r = _pt(lm, _BROW_R)
    eye_mid_l = (_pt(lm, _EYE_TOP_L)[1] + _pt(lm, _EYE_BOT_L)[1]) * 0.5
    eye_mid_r = (_pt(lm, _EYE_TOP_R)[1] + _pt(lm, _EYE_BOT_R)[1]) * 0.5
    brow_raise = ((eye_mid_l - brow_l[1]) + (eye_mid_r - brow_r[1])) * 0.5 / scale

    brow_furrow = _dist(_pt(lm, _BROW_INNER_L), _pt(lm, _BROW_INNER_R)) / scale

    def ear(top: int, bot: int, side_a: int, side_b: int) -> float:
        v = _dist(_pt(lm, top), _pt(lm, bot))
        h = _dist(_pt(lm, side_a), _pt(lm, side_b))
        return v / (h + 1e-6)

    eye_open = (ear(_EYE_TOP_L, _EYE_BOT_L, _EYE_OUT_L, _EYE_INNER_L) + ear(_EYE_TOP_R, _EYE_BOT_R, _EYE_OUT_R, _EYE_INNER_R)) * 0.5

    nose_upper = _dist(nose, lip_u) / scale

    return {
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "smile_arch": smile_arch,
        "brow_raise": brow_raise,
        "brow_furrow": brow_furrow,
        "eye_open": eye_open,
        "nose_upper": nose_upper,
    }


def prepare_scoring_features(
    raw: Dict[str, float], baseline: Optional[Dict[str, float]]
) -> Tuple[Dict[str, float], str]:
    """Calibrated mode uses deltas + mouth width ratio; raw uses absolutes (less accurate)."""
    if baseline is None:
        return {
            "mouth_open": raw["mouth_open"],
            "smile_arch": raw["smile_arch"],
            "brow_raise": raw["brow_raise"],
            "eye_open": raw["eye_open"],
            "nose_upper": raw["nose_upper"],
            "brow_furrow": raw["brow_furrow"],
            "mouth_width": raw["mouth_width"],
        }, "raw"
    bmw = baseline["mouth_width"] + 1e-6
    return {
        "mouth_open": raw["mouth_open"] - baseline["mouth_open"],
        "smile_arch": raw["smile_arch"] - baseline["smile_arch"],
        "brow_raise": raw["brow_raise"] - baseline["brow_raise"],
        "eye_open": raw["eye_open"] - baseline["eye_open"],
        "nose_upper": raw["nose_upper"] - baseline["nose_upper"],
        "brow_furrow_delta": raw["brow_furrow"] - baseline["brow_furrow"],
        "mw_rel": raw["mouth_width"] / bmw,
    }, "calibrated"


def _inhibit(scores: Dict[str, float]) -> None:
    """Reduce common confusions (happy vs surprised, etc.)."""
    h, s = scores["happy"], scores["surprised"]
    if h > 0.2 and s > 0.2:
        scores["surprised"] *= max(0.35, 1.0 - h * 1.1)
    if s > 0.28:
        scores["happy"] *= max(0.45, 1.0 - s * 0.75)
    a, g = scores["angry"], scores["disgust"]
    if a > 0.25 and g > 0.25:
        scores["disgust"] *= max(0.5, 1.0 - a * 0.8)


def score_posed_expressions(f: Dict[str, float], mode: str) -> Dict[str, float]:
    """Heuristic scores; use mode=='calibrated' after pressing C for best accuracy."""
    if mode == "calibrated":
        mo = f["mouth_open"]
        sa = f["smile_arch"]
        br = f["brow_raise"]
        eo = f["eye_open"]
        nu = f["nose_upper"]
        bfd = f["brow_furrow_delta"]
        mwrel = f["mw_rel"]
        happy = min(
            1.0,
            max(0.0, sa * 55.0) * 0.42
            + max(0.0, (mwrel - 1.0) * 12.0) * 0.38
            + max(0.0, 0.12 - mo * 6.0) * 0.2,
        )
        sad = min(
            1.0,
            max(0.0, -sa * 50.0) * 0.48
            + max(0.0, (1.0 - mwrel) * 10.0) * 0.32
            + max(0.0, mo * 8.0) * 0.2,
        )
        surprised = min(
            1.0,
            max(0.0, mo * 14.0) * 0.48
            + max(0.0, br * 22.0) * 0.32
            + max(0.0, eo * 18.0) * 0.2,
        )
        angry = min(
            1.0,
            max(0.0, -bfd * 42.0) * 0.48
            + max(0.0, -br * 28.0 + 0.15) * 0.28
            + max(0.0, -sa * 35.0 + 0.12) * 0.24,
        )
        disgust = min(
            1.0,
            max(0.0, -nu * 32.0) * 0.48
            + max(0.0, 0.2 - mo * 6.0) * 0.28
            + max(0.0, bfd * 18.0) * 0.24,
        )
    else:
        mo, sa, br, eo, nu, bf, mw = (
            f["mouth_open"],
            f["smile_arch"],
            f["brow_raise"],
            f["eye_open"],
            f["nose_upper"],
            f["brow_furrow"],
            f["mouth_width"],
        )
        happy = min(
            1.0,
            max(0.0, (sa - 0.018) * 40.0) * 0.42
            + max(0.0, (mw - 0.44) * 6.0) * 0.38
            + max(0.0, 0.12 - mo * 5.0) * 0.2,
        )
        sad = min(
            1.0,
            max(0.0, (0.012 - sa) * 45.0) * 0.48
            + max(0.0, (0.4 - mw) * 7.0) * 0.32
            + max(0.0, mo * 6.0) * 0.2,
        )
        surprised = min(
            1.0,
            max(0.0, (mo - 0.05) * 10.0) * 0.48
            + max(0.0, (br - 0.14) * 8.0) * 0.32
            + max(0.0, (eo - 0.28) * 7.0) * 0.2,
        )
        angry = min(
            1.0,
            max(0.0, (0.52 - bf) * 9.0) * 0.48
            + max(0.0, (0.12 - br) * 10.0) * 0.28
            + max(0.0, (0.01 - sa) * 35.0) * 0.24,
        )
        disgust = min(
            1.0,
            max(0.0, (0.43 - nu) * 10.0) * 0.48
            + max(0.0, (0.42 - mo) * 5.0) * 0.28
            + max(0.0, (bf - 0.48) * 8.0) * 0.24,
        )

    scores = {
        "happy": happy,
        "sad": sad,
        "surprised": surprised,
        "angry": angry,
        "disgust": disgust,
    }
    _inhibit(scores)
    peak = max(scores.values())
    scores["neutral"] = max(0.12, min(0.92, 1.08 - peak * 1.15))
    return scores


def smooth_scores(prev: Optional[Dict[str, float]], new: Dict[str, float], alpha: float = 0.86) -> Dict[str, float]:
    if prev is None:
        return dict(new)
    return {k: alpha * prev.get(k, 0.0) + (1.0 - alpha) * new[k] for k in new}


def _softmax_probs(scores: Dict[str, float], temperature: float = 0.2) -> Dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp((v - m) / max(temperature, 1e-6)) for k, v in scores.items()}
    s = sum(exps.values())
    return {k: exps[k] / s for k in exps}


def pick_expression(
    scores: Dict[str, float],
    prev_best: Optional[str],
    switch_margin: float = 0.07,
) -> Tuple[str, float]:
    """Softmax confidence + hysteresis so labels do not flicker."""
    if prev_best is not None and prev_best not in scores:
        prev_best = None
    probs = _softmax_probs(scores, temperature=0.19)
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    best, top_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    if (
        prev_best is not None
        and best != prev_best
        and (top_p - second_p) < switch_margin
    ):
        return prev_best, probs[prev_best]
    return best, top_p


def draw_fps_hud(
    image,
    fps: float,
    x: int = 10,
    y: int = 10,
    box_w: int = 168,
    box_h: int = 42,
    bg_alpha: float = 0.45,
) -> None:
    """Semi-transparent rectangle + white FPS text (mutates image in place)."""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, image, 1.0 - bg_alpha, 0, image)
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (x + 10, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_expression_hud(
    image,
    label: str,
    conf: float,
    baseline_set: bool,
    x: int = 10,
    y: int = 58,
    box_w: int = 320,
    box_h: int = 52,
    bg_alpha: float = 0.45,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, image, 1.0 - bg_alpha, 0, image)
    line1 = f"Pose: {label} ({conf * 100:.0f}%)"
    line2 = (
        "Baseline: on — much better accuracy"
        if baseline_set
        else "Press C: neutral face for calibration (recommended)"
    )
    cv2.putText(
        image,
        line1,
        (x + 10, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 255, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        line2,
        (x + 10, y + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera — try cv2.VideoCapture(1) or another index.")

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_t = time.perf_counter()
    fps_smooth = 0.0
    fps_alpha = 0.92

    baseline: Optional[Dict[str, float]] = None
    score_smooth: Optional[Dict[str, float]] = None
    stable_expr: Optional[str] = None
    display_label, display_conf = "—", 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror (selfie-style)
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)
            rgb.flags.writeable = True

            annotated = frame
            first_face = (
                face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            )

            if first_face is not None:
                raw = compute_posed_expression_features(first_face)
                if raw is not None:
                    fadj, mode = prepare_scoring_features(raw, baseline)
                    scores = score_posed_expressions(fadj, mode)
                    score_smooth = smooth_scores(score_smooth, scores, alpha=0.86)
                    display_label, display_conf = pick_expression(
                        score_smooth, stable_expr, switch_margin=0.07
                    )
                    stable_expr = display_label
            else:
                display_label, display_conf = "—", 0.0
                stable_expr = None

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        HAND_LANDMARK_SPEC,
                        HAND_CONNECTION_SPEC,
                    )

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            inst_fps = 1.0 / dt
            fps_smooth = fps_alpha * fps_smooth + (1.0 - fps_alpha) * inst_fps if fps_smooth > 0 else inst_fps

            draw_fps_hud(annotated, fps_smooth)
            draw_expression_hud(
                annotated,
                display_label.capitalize(),
                display_conf,
                baseline_set=baseline is not None,
            )

            cv2.imshow("Face mesh + hands (C baseline, Q quit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and first_face is not None:
                raw = compute_posed_expression_features(first_face)
                baseline = dict(raw) if raw is not None else None
                score_smooth = None
                stable_expr = None
    finally:
        face_mesh.close()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
