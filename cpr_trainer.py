import cv2
import mediapipe as mp
import time
import numpy as np
import math
import csv

# ───────────────────────── CONFIG ─────────────────────────
TARGET_RATE_MIN   = 100
TARGET_RATE_MAX   = 120
TARGET_DEPTH_MSG  = "Depth: Push hard (5-6 cm)"

# colours
CHEST_TARGET_COLOR_CENTERED = (  0, 255,   0)
CHEST_TARGET_COLOR_OFF      = (  0, 165, 255)
AR_ARROW_COLOR              = (255, 255,   0)
METRONOME_COLOR_IDLE        = (100, 100, 100)
METRONOME_COLOR_BEAT        = (  0, 255,   0)
FEEDBACK_COLOR_GOOD         = (  0, 255,   0)
FEEDBACK_COLOR_WARN         = (  0, 165, 255)
FEEDBACK_COLOR_ERR          = (  0,   0, 255)
INFO_COLOR                  = (255, 255,   0)
WARNING_COLOR               = ( 50,  50, 255)

# metronome / CPR logic
BPM                   = 110
BEAT_PERIOD           = 60.0 / BPM
COMPRESSIONS_PER_SET  = 30
PRE_START_COUNTDOWN   = 3                 # 3-2-1-GO
BREATH_PHASE_SECS     = 8                 # seconds to show breath instructions
SET_TARGET            = 4                 # total sets (30c + breaths) ×4

# geometry / detection
CHEST_TARGET_RADIUS      = 15
HAND_PLACEMENT_TOLERANCE = 45             # px
COMPRESSION_THRESHOLD    = 8              # Δ-y per compression

# supine text
SUPINE_STATUS_OK_MSG   = "Victim supine"
SUPINE_STATUS_WARN_MSG = "Victim NOT lying flat!"

# ──────────────────────── MediaPipe ───────────────────────
mp_drawing  = mp.solutions.drawing_utils
mp_pose     = mp.solutions.pose
mp_hands    = mp.solutions.hands
mp_styles   = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    model_complexity          = 1,
    min_detection_confidence  = 0.5,
    min_tracking_confidence   = 0.5
)

hands = mp_hands.Hands(
    max_num_hands             = 2,
    model_complexity          = 1,
    min_detection_confidence  = 0.5,
    min_tracking_confidence   = 0.5
)

# ───────────────────────── Audio (pygame) ─────────────────
try:
    import pygame
    pygame.mixer.init()
    CLICK_WAV = pygame.mixer.Sound("click.wav")
except Exception:
    CLICK_WAV = None        # silent fallback

# ───────────────────────── Camera ─────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Web-cam not found")

# ───────────────────────── State vars ─────────────────────
compression_times     = []
current_rate          = 0
last_hand_y           = None
is_compressing        = False

set_counter           = 0                     # which 30-compression block
cycle_active          = False                 # inside a compression set
compressions_left     = COMPRESSIONS_PER_SET  # countdown within a set

pre_count_active      = False
pre_count_start_time  = None

breath_phase_active   = False
breath_start_time     = None

next_beat_time        = None
metronome_flash       = False

# breath instructions (multi-line)
breath_msg_lines = [
    "GIVE 2 BREATHS",
    "",
    "1. Open airway (head-tilt / chin-lift).",
    "2. Pinch nose, seal mouth, give 1-second breath → chest rise.",
    "3. Let air exit, then give 2nd breath."
]

# ───────────────────────── Helper fns ─────────────────────
def get_xy(lms, idx, w, h):
    try:
        pt = lms[idx.value]
        if getattr(pt, "visibility", 1.0) > .5:
            return int(pt.x * w), int(pt.y * h)
    except Exception:
        pass
    return None, None

def distance(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1]) if None not in (*p, *q) else 1e9

def victim_supine(lms, w, h):
    ls = get_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
    rs = get_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    lh = get_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,       w, h)
    rh = get_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w, h)
    if None in (*ls, *rs, *lh, *rh):
        return False
    torso = ((lh[0]+rh[0])//2 - (ls[0]+rs[0])//2,
             (lh[1]+rh[1])//2 - (ls[1]+rs[1])//2)
    ang = abs(math.degrees(math.atan2(torso[1], torso[0])))
    return ang <= 30 or ang >= 60

def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return math.degrees(math.acos(max(-1, min(1, num/den))))

def elbow_angle(lms, side, w, h):
    sh = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER"), w, h)
    el = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ELBOW"),    w, h)
    wr = get_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_WRIST"),    w, h)
    if None in (*sh, *el, *wr):
        return None
    v1, v2 = (sh[0]-el[0], sh[1]-el[1]), (wr[0]-el[0], wr[1]-el[1])
    return angle_between(v1, v2)

def draw_labels(img, vic, resc):
    if vic:
        cv2.circle(img, vic, 24, (255,255,255), 2)
        cv2.putText(img, "Victim",  (vic[0]-30, vic[1]-30),
                    cv2.FONT_HERSHEY_DUPLEX, .6, (255,255,255), 2)
    if resc:
        cv2.circle(img, resc, 24, (255,0,255), 2)
        cv2.putText(img, "Rescuer", (resc[0]-35, resc[1]-30),
                    cv2.FONT_HERSHEY_DUPLEX, .6, (255,0,255), 2)

# ───────────────────────── Main loop ─────────────────────
print("Running – press ‘q’ to quit")

while cap.isOpened():

    ok, frame = cap.read()
    if not ok:
        break

    h, w, _ = frame.shape
    frame   = cv2.flip(frame, 1)

    # ------ inference ------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_r  = pose.process(rgb)
    hands_r = hands.process(rgb)
    rgb.flags.writeable = True
    frame   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    victim_pt      = None
    resc_hand      = None
    hands_centered = False
    posture_ok     = False
    supine_ok      = False

    hand_msg    = "Looking for Victim/Hands"
    posture_msg = "Posture: Unclear"
    tgt_color   = CHEST_TARGET_COLOR_OFF

    # ---------------- victim ----------------
    if pose_r.pose_landmarks:
        lms = pose_r.pose_landmarks.landmark
        supine_ok = victim_supine(lms, w, h)

        ls = get_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)
        rs = get_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        lh = get_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,       w, h)
        rh = get_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w, h)

        if None not in (*ls, *rs, *lh, *rh):
            sh_mid  = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)

            victim_pt = ( int(sh_mid[0] + 0.33*(hip_mid[0]-sh_mid[0])),
                          int(sh_mid[1] + 0.33*(hip_mid[1]-sh_mid[1])) )
            hand_msg  = "Show Rescuer Hands"
        else:
            hand_msg  = "Show Victim Chest Area"
    else:
        lms = None
        hand_msg = "Cannot find Victim"

    # --------------- rescuer hand --------------
    if victim_pt and supine_ok and hands_r.multi_hand_landmarks:

        vic_wr = [ get_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST,  w, h),
                   get_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, w, h) ] if lms else []

        best_d = 1e9
        for hd in hands_r.multi_hand_landmarks:

            wrist = ( int(hd.landmark[0].x * w),
                      int(hd.landmark[0].y * h) )

            if any(distance(wrist, vw) < 40 for vw in vic_wr if vw[0]):
                continue    # skip victim hand

            mp_drawing.draw_landmarks(
                frame, hd, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style() )

            for lm in hd.landmark:
                pt = ( int(lm.x * w), int(lm.y * h) )
                d  = distance(pt, victim_pt)
                if d < best_d:
                    best_d, resc_hand = d, pt

        if resc_hand:
            cv2.circle(frame, resc_hand, 5, (0,255,255), -1)

            if best_d < HAND_PLACEMENT_TOLERANCE:
                hands_centered = True
                tgt_color      = CHEST_TARGET_COLOR_CENTERED
                hand_msg       = "Hands Centered"
            else:
                hand_msg       = "Move Hands Closer to Target"

    # ---------------- posture -----------------
    if lms and resc_hand:
        lw   = get_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST,  w, h)
        rw   = get_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        side = "LEFT" if distance(resc_hand, lw) < distance(resc_hand, rw) else "RIGHT"

        ang  = elbow_angle(lms, side, w, h)
        if ang:
            posture_ok  = ang >= 160
            posture_msg = "Posture: Good" if posture_ok else f"Posture: Straighten {side.lower()} arm"

    # ---------------- state machine ------------
    now = time.time()

    # phase: waiting for readiness
    if not (cycle_active or pre_count_active or breath_phase_active):
        ready = hands_centered and posture_ok and supine_ok

        if ready and set_counter < SET_TARGET:
            pre_count_active     = True
            pre_count_start_time = now

    # pre-count → start compressions
    if pre_count_active and now - pre_count_start_time >= PRE_START_COUNTDOWN:
        pre_count_active     = False
        cycle_active         = True
        compressions_left    = COMPRESSIONS_PER_SET
        next_beat_time       = now

    # metronome
    if cycle_active and next_beat_time:
        metronome_flash = now >= next_beat_time
        if metronome_flash:
            next_beat_time += BEAT_PERIOD
            if CLICK_WAV:
                CLICK_WAV.play()
    else:
        metronome_flash = False

    # compression detection
    if cycle_active and resc_hand:
        y = resc_hand[1]
        if last_hand_y is not None:
            dy = y - last_hand_y

            if dy > COMPRESSION_THRESHOLD and not is_compressing:
                is_compressing = True

            elif dy < -COMPRESSION_THRESHOLD and is_compressing:
                is_compressing = False
                compressions_left -= 1

                if CLICK_WAV is None:
                    metronome_flash = True

                if compressions_left <= 0:
                    cycle_active   = False
                    next_beat_time = None
                    set_counter   += 1

                    if set_counter < SET_TARGET:
                        breath_phase_active = True
                        breath_start_time   = now
        last_hand_y = y
    else:
        last_hand_y    = None
        is_compressing = False

    # breath phase timeout → wait for next readiness
    if breath_phase_active and now - breath_start_time >= BREATH_PHASE_SECS:
        breath_phase_active = False

    # ---------------- UI drawing --------------
    if victim_pt:
        cv2.circle(frame, victim_pt, CHEST_TARGET_RADIUS, tgt_color, -1)
        cv2.circle(frame, victim_pt, CHEST_TARGET_RADIUS+5, tgt_color, 2)

    # general info lines
    cv2.putText(frame, f"Placement: {hand_msg}", (10,60),  cv2.FONT_HERSHEY_DUPLEX,
                1.3, FEEDBACK_COLOR_GOOD if hands_centered else FEEDBACK_COLOR_WARN, 2)

    sup_txt = SUPINE_STATUS_OK_MSG if supine_ok else SUPINE_STATUS_WARN_MSG
    cv2.putText(frame, sup_txt, (10,100), cv2.FONT_HERSHEY_DUPLEX,
                1.3, FEEDBACK_COLOR_GOOD if supine_ok else FEEDBACK_COLOR_ERR, 2)

    cv2.putText(frame, f"Rate: {current_rate:.0f} /min", (10,140),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, INFO_COLOR, 2)

    cv2.putText(frame, posture_msg, (10,180), cv2.FONT_HERSHEY_DUPLEX,
                1.3, FEEDBACK_COLOR_GOOD if posture_ok else FEEDBACK_COLOR_WARN, 2)

    cv2.putText(frame, TARGET_DEPTH_MSG, (10,220), cv2.FONT_HERSHEY_DUPLEX,
                1.3, FEEDBACK_COLOR_ERR, 2)

    # phase-specific panels
    # No longer using fixed status_y_start

    if pre_count_active:
        cd = int(PRE_START_COUNTDOWN - (now - pre_count_start_time)) + 1
        start_text = f"Starting in {cd}"
        font_scale = 1.5
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        # Center horizontally and vertically
        text_x = (w - text_w) // 2
        text_y = (h - text_h) // 2 # Vertical centering
        cv2.putText(frame, start_text, (text_x, text_y + text_h), # Draw using bottom-left corner
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, INFO_COLOR, thickness)

    elif cycle_active:
        compressions_text = f"Compressions left: {compressions_left}"
        font_scale = 1.5
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(compressions_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        # Center horizontally and vertically
        text_x = (w - text_w) // 2
        text_y = (h - text_h) // 2 # Vertical centering
        cv2.putText(frame, compressions_text, (text_x, text_y + text_h), # Draw using bottom-left corner
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, INFO_COLOR, thickness)

    elif breath_phase_active:
        font_scale = 1.3
        thickness = 2
        line_height = 40
        # Calculate total height of the breath instruction block
        total_block_height = len(breath_msg_lines) * line_height
        # Calculate starting Y to center the block vertically
        y_block_start = (h - total_block_height) // 2

        for i, line in enumerate(breath_msg_lines):
            current_y = y_block_start + i * line_height + line_height # Use bottom-left for putText
            if i == 0: # Center the main "GIVE 2 BREATHS" title
                (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = (w - text_w) // 2
            else: # Keep subsequent lines left-aligned
                text_x = 10
            cv2.putText(frame, line, (text_x, current_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, INFO_COLOR, thickness)

    elif set_counter >= SET_TARGET:
        complete_text = "4 sets complete – training cycle finished"
        font_scale = 1.5
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(complete_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        # Center horizontally and vertically
        text_x = (w - text_w) // 2
        text_y = (h - text_h) // 2 # Vertical centering
        cv2.putText(frame, complete_text, (text_x, text_y + text_h), # Draw using bottom-left corner
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, INFO_COLOR, thickness)

    # metronome circle
    metro_col = METRONOME_COLOR_BEAT if metronome_flash else METRONOME_COLOR_IDLE
    cv2.circle(frame, (w-50,50), 20, metro_col, -1)

    # bottom disclaimer
    cv2.putText(frame, "TRAINING AID ONLY", (10,h-15),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, WARNING_COLOR, 2)

    # guidance arrow
    if victim_pt and resc_hand and not hands_centered:
        cv2.arrowedLine(frame, resc_hand, victim_pt, AR_ARROW_COLOR, 3, tipLength=.25)

    draw_labels(frame, victim_pt, resc_hand)

    cv2.imshow("CPR Trainer", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ─────────────────────── cleanup ────────────────────────
pose.close()
hands.close()
cap.release()
cv2.destroyAllWindows()
try:
    pygame.mixer.quit()
except Exception:
    pass
