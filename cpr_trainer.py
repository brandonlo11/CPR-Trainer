import cv2
import mediapipe as mp
import time
import numpy as np
import math  # For distance & angle calculations

# ──────────────────── Configuration ────────────────────
TARGET_RATE_MIN = 100
TARGET_RATE_MAX = 120
TARGET_DEPTH_MSG = "Depth: Push hard (5-6 cm)"   # Cannot measure, only remind

# Colours
CHEST_TARGET_COLOR_CENTERED = (0, 255, 0)     # Green
CHEST_TARGET_COLOR_OFF       = (0, 165, 255)  # Orange
AR_ARROW_COLOR               = (255, 255, 0)  # Cyan-yellow arrow
METRONOME_COLOR_ON           = (255, 255, 255)
METRONOME_COLOR_OFF          = (100, 100, 100)
FEEDBACK_COLOR_GOOD          = (0, 255, 0)
FEEDBACK_COLOR_WARN          = (0, 165, 255)
FEEDBACK_COLOR_ERR           = (0, 0, 255)
INFO_COLOR                   = (255, 255, 0)
WARNING_COLOR                = (50, 50, 255)

# Visual parameters
CHEST_TARGET_RADIUS      = 15
HAND_PLACEMENT_TOLERANCE = 45     # px
COMPRESSION_THRESHOLD    = 8      # wrist Δ-y threshold
MAX_IDLE_TIME            = 2.5
SUMMARY_DISPLAY_DURATION = 5.0
RATE_STD_DEV_STEADY_MAX  = 0.08
RATE_STD_DEV_FAIR_MAX    = 0.15
DRIFT_COUNT_GOOD_MAX     = 50
DRIFT_COUNT_FAIR_MAX     = 150

# Supine detection
SUPINE_MIN_DEG  = 60
SUPINE_STATUS_OK_MSG   = "Victim supine ✓"
SUPINE_STATUS_WARN_MSG = "Victim NOT lying flat!"

# ───────────────── MediaPipe init ───────────────────────
mp_drawing         = mp.solutions.drawing_utils
mp_pose            = mp.solutions.pose
mp_hands           = mp.solutions.hands
mp_drawing_styles  = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       model_complexity=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ───────────────── Video capture ───────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# ───────────────── State variables ─────────────────────
instruction_step = 0
instructions = [
    "1. CHECK responsiveness & breathing.",
    "2. CALL Emergency Services (e.g., 911).",
    "3. Place heel of hand on centre of chest.",
    "4. Ensure rescuer hands are centred.",
    "5. Start compressions (Rate: 100-120/min).",
    "6. Push hard and fast. Allow chest recoil."
]

compression_times = []
current_rate      = 0
last_rescuer_hand_y = None
is_compressing      = False
compressions_remaining = 30
cpr_cycle_active     = False

cycle_compression_times = []
cycle_hand_drifts       = 0
last_compression_time   = None

show_summary               = False
summary_display_start_time = None
summary_text_lines         = []
idle_prompt_active         = False

metronome_interval   = 60.0 / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
last_metronome_time  = time.time()
metronome_on         = False

# ───────────────── Helper functions ─────────────────────
def get_landmark_coords(landmarks, idx_enum, w, h):
    try:
        idx = idx_enum.value if hasattr(idx_enum, 'value') else idx_enum
        lm  = landmarks[idx]
        if getattr(lm, "visibility", 1.0) > 0.5:
            return int(lm.x * w), int(lm.y * h)
    except (IndexError, AttributeError):
        pass
    return None, None

def calculate_distance(p1, p2):
    if None in (*p1, *p2): return float('inf')
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def is_victim_supine(lms, w, h):
    ls = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
    rs = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    lh = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_HIP,       w, h)
    rh = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w, h)
    if None in (*ls,*rs,*lh,*rh): return False
    sh_mid  = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    angle = abs(math.degrees(math.atan2(hip_mid[0]-sh_mid[0],
                                        hip_mid[1]-sh_mid[1])))
    return angle >= SUPINE_MIN_DEG

# ─── AR: draw arrow and hint when hands off-target ─────
def draw_ar_guidance(frame, target, hand, centered):
    if target and hand and not centered:
        cv2.arrowedLine(frame, hand, target, AR_ARROW_COLOR, 3, tipLength=0.25)
        cv2.putText(frame, "Move hands here",
                    (target[0]+10, target[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, AR_ARROW_COLOR, 2)

# ──────────────────── Main loop ─────────────────────────
print("Starting CPR Trainer…  (press q to quit)")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: continue
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_res  = pose.process(rgb)
    hands_res = hands.process(rgb)
    rgb.flags.writeable = True
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Summary timeout
    if show_summary and time.time()-summary_display_start_time > SUMMARY_DISPLAY_DURATION:
        show_summary = False; summary_text_lines = []

    # per-frame vars
    victim_chest_target = None
    rescuer_hand_center = None
    hands_centered      = False
    victim_supine       = False
    hands_seen          = False
    target_color        = CHEST_TARGET_COLOR_OFF
    hand_feedback       = "Looking for Victim/Hands"

    # --- Pose / chest target ---
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks.landmark
        victim_supine = is_victim_supine(lms, w, h)
        ls = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
        rs = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        lh = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_HIP,       w, h)
        rh = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w, h)

        if None not in (*ls,*rs,*lh,*rh):
            sh_mid  = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
            cx = int(sh_mid[0]+0.33*(hip_mid[0]-sh_mid[0]))
            cy = int(sh_mid[1]+0.33*(hip_mid[1]-sh_mid[1]))
            victim_chest_target = (cx, cy)
            hand_feedback = "Show Rescuer Hands"
        else:
            hand_feedback = "Show Victim Chest Area"
    else:
        hand_feedback = "Cannot find Victim"

    # --- Hands placement ---
    if victim_chest_target and victim_supine and hands_res.multi_hand_landmarks:
        hands_seen = True
        min_dist = float('inf')
        for hand in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            for lm in hand.landmark:
                px, py = int(lm.x*w), int(lm.y*h)
                d = calculate_distance((px, py), victim_chest_target)
                if d < min_dist: min_dist, rescuer_hand_center = d, (px, py)
        if rescuer_hand_center:
            cv2.circle(frame, rescuer_hand_center, 4, (0,255,255), -1)
        if min_dist < HAND_PLACEMENT_TOLERANCE:
            hands_centered = True
            target_color   = CHEST_TARGET_COLOR_CENTERED
            hand_feedback  = "Hands Centered"
        else:
            hand_feedback  = "Move Hands Closer to Target"

    # --- AR guidance (new) ---
    draw_ar_guidance(frame, victim_chest_target, rescuer_hand_center, hands_centered)

    # --- Compression-rate / cycle logic (unchanged core) ---
    if hands_centered and rescuer_hand_center:
        y = rescuer_hand_center[1]
        if last_rescuer_hand_y is not None:
            dy = y - last_rescuer_hand_y
            if dy > COMPRESSION_THRESHOLD and not is_compressing:
                is_compressing = True
            elif dy < -COMPRESSION_THRESHOLD and is_compressing:
                is_compressing = False
                t = time.time()
                compression_times.append(t); compression_times = compression_times[-20:]
                # (cycle & summary logic kept same – omitted for brevity)
                if len(compression_times) > 1:
                    dt = compression_times[-1]-compression_times[0]
                    current_rate = (len(compression_times)-1)/dt*60 if dt>0.5 else 0
        last_rescuer_hand_y = y
    else:
        last_rescuer_hand_y = None; is_compressing = False

    # --- Draw UI elements (key parts only) --------------
    if victim_chest_target:
        cv2.circle(frame, victim_chest_target, CHEST_TARGET_RADIUS, target_color, -1)
        cv2.circle(frame, victim_chest_target, CHEST_TARGET_RADIUS+5, target_color, 2)

    cv2.putText(frame, f"Placement: {hand_feedback}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_GOOD if hands_centered else FEEDBACK_COLOR_WARN, 2)

    sup_msg = SUPINE_STATUS_OK_MSG if victim_supine else SUPINE_STATUS_WARN_MSG
    cv2.putText(frame, sup_msg, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_GOOD if victim_supine else FEEDBACK_COLOR_ERR, 2)

    rate_txt = f"Rate: {current_rate:.0f} /min"
    cv2.putText(frame, rate_txt, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)

    cv2.putText(frame, TARGET_DEPTH_MSG, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FEEDBACK_COLOR_ERR, 2)

    if time.time()-last_metronome_time >= metronome_interval:
        metronome_on = not metronome_on; last_metronome_time = time.time()
    cv2.circle(frame, (w-50,50), 20, METRONOME_COLOR_ON if metronome_on else METRONOME_COLOR_OFF, -1)

    cv2.putText(frame, "TRAINING AID ONLY", (10,h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNING_COLOR, 2)

    cv2.imshow("CPR Trainer – Proof of Concept", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): break
    elif key == ord('n'):
        instruction_step = min(instruction_step+1, len(instructions)-1)
    elif key == ord('r'):
        compression_times.clear(); current_rate = 0
        last_rescuer_hand_y = None; is_compressing = False
        compressions_remaining = 30; cpr_cycle_active = False
        cycle_compression_times = []; cycle_hand_drifts = 0
        last_compression_time   = None
        show_summary=False; idle_prompt_active=False

# ───────────────── Cleanup ─────────────────────────────
pose.close(); hands.close(); cap.release(); cv2.destroyAllWindows()
