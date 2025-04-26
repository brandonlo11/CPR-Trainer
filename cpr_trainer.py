import cv2
import mediapipe as mp
import time
import numpy as np
import math
import csv

# ──────────────────── Configuration ────────────────────
TARGET_RATE_MIN  = 100
TARGET_RATE_MAX  = 120
TARGET_DEPTH_MSG = "Depth: Push hard (5-6 cm)"

# Colours
CHEST_TARGET_COLOR_CENTERED = (0, 255, 0)
CHEST_TARGET_COLOR_OFF      = (0, 165, 255)
AR_ARROW_COLOR              = (255, 255, 0)
METRONOME_COLOR_ON          = (255, 255, 255)
METRONOME_COLOR_OFF         = (100, 100, 100)
FEEDBACK_COLOR_GOOD         = (0, 255, 0)
FEEDBACK_COLOR_WARN         = (0, 165, 255)
FEEDBACK_COLOR_ERR          = (0,   0, 255)
INFO_COLOR                  = (255, 255, 0)
WARNING_COLOR               = (50,  50, 255)

# Visual parameters
CHEST_TARGET_RADIUS      = 15
HAND_PLACEMENT_TOLERANCE = 45          # px
COMPRESSION_THRESHOLD    = 8           # Δ-y for one compression

# Supine status messages
SUPINE_STATUS_OK_MSG   = "Victim supine ✓"
SUPINE_STATUS_WARN_MSG = "Victim NOT lying flat!"

# ───────────────── MediaPipe init ───────────────────────
mp_drawing        = mp.solutions.drawing_utils
mp_pose           = mp.solutions.pose
mp_hands          = mp.solutions.hands
mp_styles         = mp.solutions.drawing_styles

pose  = mp_pose.Pose(model_complexity=1,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ───────────────── Video capture ───────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# ───────────────── State vars ──────────────────────────
compression_times = []
current_rate      = 0
last_hand_y       = None
is_compressing    = False
metronome_interval = 60 / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
last_metronome_time = time.time()
metronome_on        = False

# ───────────────── Helper functions ────────────────────
def get_xy(lms, idx, w, h):
    try:
        pt = lms[idx.value]
        if getattr(pt, "visibility", 1.0) > .5:
            return int(pt.x * w), int(pt.y * h)
    except: pass
    return None, None

def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1]) if None not in (*p, *q) else 1e9

def victim_supine(lms, w, h):
    ls, rs = get_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h), \
             get_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    lh, rh = get_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,      w, h), \
             get_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP,     w, h)
    if None in (*ls, *rs, *lh, *rh): return False
    vec = ((lh[0]+rh[0])//2 - (ls[0]+rs[0])//2,
           (lh[1]+rh[1])//2 - (ls[1]+rs[1])//2)
    ang = abs(math.degrees(math.atan2(vec[1], vec[0])))
    return ang <= 30 or ang >= 60          # horizontal OR vertical torso

def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return math.degrees(math.acos(max(-1, min(1, num / den))))

def elbow_angle(lms, side, w, h):
    sh = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER"), w, h)
    el = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ELBOW"),    w, h)
    wr = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_WRIST"),    w, h)
    if None in (*sh, *el, *wr): return None
    v1, v2 = (sh[0]-el[0], sh[1]-el[1]), (wr[0]-el[0], wr[1]-el[1])
    return angle_between(v1, v2)

def draw_labels(img, victim_pt, resc_pt):
    if victim_pt:
        cv2.circle(img, victim_pt, 24, (255,255,255), 2)
        cv2.putText(img, "Victim", (victim_pt[0]-30, victim_pt[1]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
    if resc_pt:
        cv2.circle(img, resc_pt, 24, (255,0,255), 2)
        cv2.putText(img, "Rescuer", (resc_pt[0]-35, resc_pt[1]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,255), 2)

# ──────────────────── Main loop ─────────────────────────
print("CPR Trainer – press ‘q’ to quit")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)

    # --- detect pose & hands ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_res = pose.process(rgb)
    hands_res = hands.process(rgb)
    rgb.flags.writeable = True
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    victim_center = resc_hand_pt = None
    hand_msg = "Looking for Victim/Hands"
    posture_msg = "Posture: Unclear"
    posture_ok  = False
    hands_centered = False
    target_col = CHEST_TARGET_COLOR_OFF
    supine_ok  = False

    # --------- Victim detection -----------
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks.landmark
        supine_ok = victim_supine(lms, w, h)
        # chest point (1/3 of way from shoulders to hips)
        ls, rs = get_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h), \
                 get_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        lh, rh = get_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,       w, h), \
                 get_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w, h)
        if None not in (*ls,*rs,*lh,*rh):
            sh_mid  = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
            victim_center = (int(sh_mid[0] + 0.33*(hip_mid[0]-sh_mid[0])),
                             int(sh_mid[1] + 0.33*(hip_mid[1]-sh_mid[1])))
            hand_msg = "Show Rescuer Hands"
        else:
            hand_msg = "Show Victim Chest Area"
    else:
        lms = None
        hand_msg = "Cannot find Victim"

    # --------- Rescuer hand detection -----------
    if victim_center and supine_ok and hands_res.multi_hand_landmarks:
        # exclude hands whose wrist lies on victim wrists (≤40 px)
        vic_wrists = [get_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST,  w, h),
                      get_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)] if lms else []
        best_d = 1e9
        for h_land in hands_res.multi_hand_landmarks:
            wrist = h_land.landmark[0]
            wrist_px = (int(wrist.x*w), int(wrist.y*h))
            if any(distance(wrist_px, vw) < 40 for vw in vic_wrists if vw[0]):  # skip victim hand
                continue
            mp_drawing.draw_landmarks(frame, h_land, mp_hands.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())
            for lm in h_land.landmark:
                p = (int(lm.x*w), int(lm.y*h))
                d = distance(p, victim_center)
                if d < best_d:
                    best_d, resc_hand_pt = d, p
        if resc_hand_pt:
            cv2.circle(frame, resc_hand_pt, 5, (0,255,255), -1)
            if best_d < HAND_PLACEMENT_TOLERANCE:
                hands_centered = True
                target_col = CHEST_TARGET_COLOR_CENTERED
                hand_msg   = "Hands Centered"
            else:
                hand_msg   = "Move Hands Closer to Target"

    # ---------- Posture evaluation ----------
    if lms and resc_hand_pt:
        # decide nearer wrist side
        left_wrist  = get_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST,  w, h)
        right_wrist = get_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        side_sel = "LEFT" if distance(resc_hand_pt, left_wrist) < \
                            distance(resc_hand_pt, right_wrist) else "RIGHT"
        ang = elbow_angle(lms, side_sel, w, h)
        if ang:
            posture_ok  = ang >= 160
            posture_msg = "Posture: Good" if posture_ok else f"Posture: Straighten {side_sel.lower()} arm"

    # ---------- Compression-rate ----------
    if hands_centered and resc_hand_pt:
        y = resc_hand_pt[1]
        if last_hand_y is not None:
            dy = y - last_hand_y
            if dy > COMPRESSION_THRESHOLD and not is_compressing:
                is_compressing = True
            elif dy < -COMPRESSION_THRESHOLD and is_compressing:
                is_compressing = False
                t = time.time()
                compression_times.append(t); compression_times = compression_times[-20:]
                with open("cpr_log.csv", "a", newline="") as f:
                    csv.writer(f).writerow([t, int(posture_ok)])
                if len(compression_times) > 1:
                    dt = compression_times[-1] - compression_times[0]
                    if dt > 0.5:
                        current_rate = (len(compression_times)-1) / dt * 60
        last_hand_y = y
    else:
        last_hand_y = None
        is_compressing = False

    # ---------- AR overlays ----------
    if victim_center and resc_hand_pt and not hands_centered:
        cv2.arrowedLine(frame, resc_hand_pt, victim_center, AR_ARROW_COLOR, 3, tipLength=.25)
    draw_labels(frame, victim_center, resc_hand_pt)

    # ---------- UI text ----------
    if victim_center:
        cv2.circle(frame, victim_center, CHEST_TARGET_RADIUS, target_col, -1)
        cv2.circle(frame, victim_center, CHEST_TARGET_RADIUS+5, target_col, 2)

    cv2.putText(frame, f"Placement: {hand_msg}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_GOOD if hands_centered else FEEDBACK_COLOR_WARN, 2)

    sup_line = SUPINE_STATUS_OK_MSG if supine_ok else SUPINE_STATUS_WARN_MSG
    cv2.putText(frame, sup_line, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_GOOD if supine_ok else FEEDBACK_COLOR_ERR, 2)

    cv2.putText(frame, f"Rate: {current_rate:.0f} /min", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)

    cv2.putText(frame, posture_msg, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_GOOD if posture_ok else FEEDBACK_COLOR_WARN, 2)

    cv2.putText(frame, TARGET_DEPTH_MSG, (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_ERR, 2)

    # metronome
    if time.time() - last_metronome_time >= metronome_interval:
        metronome_on = not metronome_on
        last_metronome_time = time.time()
    cv2.circle(frame, (w-50,50), 20,
               METRONOME_COLOR_ON if metronome_on else METRONOME_COLOR_OFF, -1)

    cv2.putText(frame, "TRAINING AID ONLY", (10,h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNING_COLOR, 2)

    cv2.imshow("CPR Trainer – Proof of Concept", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ───────────────── Cleanup ─────────────────────────────
pose.close(); hands.close(); cap.release(); cv2.destroyAllWindows()
