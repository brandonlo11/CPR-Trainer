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
RESCUER_HAND_COLOR           = (255, 0, 255)  # Magenta
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
COMPRESSION_THRESHOLD    = 8      # wrist Δ-y to count a comp.

# ─── Supine-detection parameters ────────────────────────
SUPINE_MIN_DEG            = 60    # ≥ this angle ⇒ lying flat
SUPINE_STATUS_OK_MSG      = "Victim supine ✓"
SUPINE_STATUS_WARN_MSG    = "Victim NOT lying flat!"

# ───────────────── MediaPipe initialisation ─────────────
mp_drawing         = mp.solutions.drawing_utils
mp_pose            = mp.solutions.pose
mp_hands           = mp.solutions.hands
mp_drawing_styles  = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ───────────────── Video capture ────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# ───────────────── State variables ──────────────────────
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

hand_placement_feedback = "Looking for Victim/Hands"
hands_centered          = False

metronome_interval   = 60.0 / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
last_metronome_time  = time.time()
metronome_on         = False

# ───────────────── Helper functions ─────────────────────
def get_landmark_coords(landmarks, landmark_index_or_enum, w, h):
    try:
        idx = (landmark_index_or_enum.value
               if hasattr(landmark_index_or_enum, 'value')
               else landmark_index_or_enum)
        lm  = landmarks[idx]
        vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
        if vis > 0.5:
            return int(lm.x * w), int(lm.y * h)
    except (IndexError, AttributeError):
        pass
    return None, None

def calculate_distance(p1, p2):
    if None in (*p1, *p2):
        return float('inf')
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_victim_supine(pose_landmarks, w, h):
    ls = get_landmark_coords(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
    rs = get_landmark_coords(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    lh = get_landmark_coords(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP,      w, h)
    rh = get_landmark_coords(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP,     w, h)
    if None in (*ls, *rs, *lh, *rh):
        return False
    sh_mid  = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    dx, dy  = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
    if dx == dy == 0:
        return False
    angle_deg = abs(math.degrees(math.atan2(dx, dy)))
    return angle_deg >= SUPINE_MIN_DEG

# ──────────────────── Main loop ─────────────────────────
print("Starting CPR Trainer…\nTHIS IS A TRAINING AID ONLY.")
while cap.isOpened():
    ok, image = cap.read()
    if not ok: continue

    h, w, _ = image.shape
    image = cv2.flip(image, 1)
    rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_results  = pose.process(rgb)
    hands_results = hands.process(rgb)
    rgb.flags.writeable = True
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ─── Per-frame resets ───────────────────────────────
    victim_chest_target = None
    rescuer_hand_center = None
    hand_placement_feedback = "Looking for Victim/Hands"
    hands_centered          = False
    current_target_color    = CHEST_TARGET_COLOR_OFF
    victim_supine           = False
    hands_detected_this_frame = False

    # ─── Pose: chest target & supine check ───────────────
    if pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks.landmark
        victim_supine = is_victim_supine(lms, w, h)

        ls = get_landmark_coords(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
        rs = get_landmark_coords(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        if ls[0] and rs[0]:
            victim_chest_target = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            hand_placement_feedback = "Show Rescuer Hands"
        else:
            hand_placement_feedback = "Show Victim Chest Area"
    else:
        hand_placement_feedback = "Cannot find Victim"

    # ─── NEW hand-placement logic ────────────────────────
    if victim_chest_target and victim_supine and hands_results.multi_hand_landmarks:
        hands_detected_this_frame = True
        min_dist = float('inf')

        for hand_idx, hand_lms in enumerate(hands_results.multi_hand_landmarks):
            # Draw entire hand
            mp_drawing.draw_landmarks(
                image, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Check **every** landmark in this hand
            for lm in hand_lms.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                dist   = calculate_distance((px, py), victim_chest_target)
                if dist < min_dist:
                    min_dist            = dist
                    rescuer_hand_center = (px, py)

            # Optional: mark the closest point per hand
            if rescuer_hand_center:
                cv2.circle(image, rescuer_hand_center, 4, (0, 255, 255), -1)

        # Decide if placement is good
        if min_dist < HAND_PLACEMENT_TOLERANCE:
            hands_centered          = True
            current_target_color    = CHEST_TARGET_COLOR_CENTERED
            hand_placement_feedback = "Hands Centered"
        else:
            hand_placement_feedback = "Move Hands Closer to Target"

    # ─── Compression-rate estimation (unchanged) ────────
    if hands_centered and rescuer_hand_center:
        y = rescuer_hand_center[1]
        if last_rescuer_hand_y is not None:
            dy = y - last_rescuer_hand_y
            if dy > COMPRESSION_THRESHOLD and not is_compressing:
                is_compressing = True
            elif dy < -COMPRESSION_THRESHOLD and is_compressing:
                is_compressing = False
                now = time.time()
                compression_times.append(now)
                compression_times = compression_times[-20:]
                if len(compression_times) > 1:
                    dt = compression_times[-1] - compression_times[0]
                    current_rate = ((len(compression_times) - 1) / dt * 60
                                    if dt > 0.5 else 0)
        last_rescuer_hand_y = y
    else:
        last_rescuer_hand_y = None
        is_compressing      = False

    # ─── Draw UI elements (unchanged) ────────────────────
    if victim_chest_target:
        cv2.circle(image, victim_chest_target, CHEST_TARGET_RADIUS, current_target_color, -1)
        cv2.circle(image, victim_chest_target, CHEST_TARGET_RADIUS + 5, current_target_color, 2)

    if instruction_step < len(instructions):
        cv2.putText(image, instructions[instruction_step], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)

    if not pose_results.pose_landmarks:
        final_feedback, placement_color = "Cannot find Victim", FEEDBACK_COLOR_ERR
    elif not victim_supine:
        final_feedback, placement_color = "Lay victim flat on back", FEEDBACK_COLOR_ERR
    elif not hands_detected_this_frame:
        final_feedback, placement_color = "Cannot find Hands", FEEDBACK_COLOR_ERR
    else:
        final_feedback = hand_placement_feedback
        placement_color = FEEDBACK_COLOR_GOOD if hands_centered else FEEDBACK_COLOR_WARN
    cv2.putText(image, f"Placement: {final_feedback}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, placement_color, 2)

    supine_msg   = SUPINE_STATUS_OK_MSG if victim_supine else SUPINE_STATUS_WARN_MSG
    supine_color = FEEDBACK_COLOR_GOOD   if victim_supine else FEEDBACK_COLOR_ERR
    cv2.putText(image, supine_msg, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, supine_color, 2)

    rate_text = f"Rate: {current_rate:.0f} /min"
    rate_color = FEEDBACK_COLOR_GOOD
    if current_rate > 0:
        if current_rate < TARGET_RATE_MIN:
            rate_text += " (Too Slow)";  rate_color = FEEDBACK_COLOR_WARN
        elif current_rate > TARGET_RATE_MAX:
            rate_text += " (Too Fast)";  rate_color = FEEDBACK_COLOR_WARN
        else:
            rate_text += " (Good)"
    cv2.putText(image, rate_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rate_color, 2)

    cv2.putText(image, TARGET_DEPTH_MSG, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, FEEDBACK_COLOR_ERR, 2)

    if time.time() - last_metronome_time >= metronome_interval:
        metronome_on = not metronome_on
        last_metronome_time = time.time()
    metro_col = METRONOME_COLOR_ON if metronome_on else METRONOME_COLOR_OFF
    cv2.circle(image, (w - 50, 50), 20, metro_col, -1)

    cv2.putText(image,
        "TRAINING AID ONLY – cannot measure depth – not for real emergencies",
        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNING_COLOR, 2)

    cv2.imshow('CPR Trainer – Proof of Concept', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        instruction_step = min(instruction_step + 1, len(instructions) - 1)
    elif key == ord('r'):
        compression_times.clear()
        current_rate = 0
        last_rescuer_hand_y = None
        is_compressing = False
        print("State reset.")

# ───────────────── Cleanup ──────────────────────────────
pose.close()
hands.close()
cap.release()
cv2.destroyAllWindows()
