import cv2
import mediapipe as mp
import time
import numpy as np
import math
import csv

# ────────────────────── CONFIG ──────────────────────
TARGET_RATE_MIN  = 100
TARGET_RATE_MAX  = 120
TARGET_DEPTH_MSG = "Depth: Push hard (5-6 cm)"

# colours
CHEST_TARGET_COLOR_CENTERED = (0, 255,   0)
CHEST_TARGET_COLOR_OFF      = (0, 165, 255)
AR_ARROW_COLOR              = (255, 255,   0)
METRONOME_COLOR_IDLE        = (100, 100, 100)   # grey
METRONOME_COLOR_BEAT        = (  0, 255,   0)   # flash green
FEEDBACK_COLOR_GOOD         = (0, 255,   0)
FEEDBACK_COLOR_WARN         = (0, 165, 255)
FEEDBACK_COLOR_ERR          = (0,   0, 255)
INFO_COLOR                  = (255, 255, 0)
WARNING_COLOR               = (50,  50, 255)

# metronome / cycle
BPM                   = 110          # mid-point of guideline range
BEAT_PERIOD           = 60.0 / BPM
COMPRESSIONS_PER_SET  = 30
PRE_START_COUNTDOWN   = 3            # “3-2-1-GO”

# geometry
CHEST_TARGET_RADIUS      = 15
HAND_PLACEMENT_TOLERANCE = 45        # px
COMPRESSION_THRESHOLD    = 8         # Δ-y for each compression

# supine text
SUPINE_STATUS_OK_MSG   = "Victim supine ✓"
SUPINE_STATUS_WARN_MSG = "Victim NOT lying flat!"

# ────────────────────── MEDIAPIPE ─────────────────────
mp_drawing, mp_pose, mp_hands, mp_styles = (
    mp.solutions.drawing_utils, mp.solutions.pose,
    mp.solutions.hands,          mp.solutions.drawing_styles)

pose  = mp_pose.Pose(model_complexity=1,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

hands = mp_hands.Hands(max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ─────────────────────── AUDIO (optional) ─────────────
try:
    import simpleaudio as sa
    CLICK_WAV = sa.WaveObject.from_wave_file("click.wav")  # short tick file
except Exception:
    CLICK_WAV = None   # silent if file or lib not present

# ────────────────────── CAMERA ────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Web-cam not found")

# ────────────────────── STATE ─────────────────────────
compression_times   = []
current_rate        = 0
last_hand_y         = None
is_compressing      = False

cycle_active        = False
compressions_left   = COMPRESSIONS_PER_SET

pre_count_active    = False
pre_count_start     = None

next_beat_time      = None
metronome_flash     = False   # one-frame flash flag

# ────────────────────── HELPERS ───────────────────────
def get_xy(lms, idx, w, h):
    try:
        pt = lms[idx.value]
        if getattr(pt, "visibility", 1.0) > .5:
            return int(pt.x * w), int(pt.y * h)
    except: pass
    return None, None

def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1]) if None not in (*p,*q) else 1e9

def victim_supine(lms, w, h):
    ls,rs = get_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER,  w,h), \
            get_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, w,h)
    lh,rh = get_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,       w,h), \
            get_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP,      w,h)
    if None in (*ls,*rs,*lh,*rh): return False
    torso = ((lh[0]+rh[0])//2 - (ls[0]+rs[0])//2,
             (lh[1]+rh[1])//2 - (ls[1]+rs[1])//2)
    ang = abs(math.degrees(math.atan2(torso[1], torso[0])))
    return ang <= 30 or ang >= 60

def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6
    return math.degrees(math.acos(max(-1, min(1, num / den))))

def elbow_angle(lms, side, w, h):
    sh = get_xy(lms, getattr(mp_pose.PoseLandmark,f"{side}_SHOULDER"),w,h)
    el = get_xy(lms, getattr(mp_pose.PoseLandmark,f"{side}_ELBOW"),   w,h)
    wr = get_xy(lms, getattr(mp_pose.PoseLandmark,f"{side}_WRIST"),   w,h)
    if None in (*sh,*el,*wr): return None
    v1,v2 = (sh[0]-el[0],sh[1]-el[1]), (wr[0]-el[0],wr[1]-el[1])
    return angle_between(v1, v2)

def draw_labels(img, vic, resc):
    if vic is not None:
        cv2.circle(img, vic, 24, (255,255,255), 2)
        cv2.putText(img,"Victim",(vic[0]-30, vic[1]-30),
                    cv2.FONT_HERSHEY_DUPLEX,.6,(255,255,255),2)
    if resc is not None:
        cv2.circle(img, resc, 24, (255,0,255), 2)
        cv2.putText(img,"Rescuer",(resc[0]-35, resc[1]-30),
                    cv2.FONT_HERSHEY_DUPLEX,.6,(255,0,255),2)

# ────────────────────── MAIN LOOP ─────────────────────
print("Running – press ‘q’ to quit")
while cap.isOpened():
    ok, frame = cap.read();  h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    # --- inference
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
    pose_r, hands_r = pose.process(rgb), hands.process(rgb)
    rgb.flags.writeable = True
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    victim_pt = resc_hand = None
    hands_centered = posture_ok = supine_ok = False
    hand_msg, posture_msg = "Looking for Victim/Hands", "Posture: Unclear"
    tgt_color = CHEST_TARGET_COLOR_OFF

    # ---------- victim  ----------
    if pose_r.pose_landmarks:
        lms = pose_r.pose_landmarks.landmark
        supine_ok = victim_supine(lms, w, h)
        ls,rs = get_xy(lms,mp_pose.PoseLandmark.LEFT_SHOULDER,w,h), \
                get_xy(lms,mp_pose.PoseLandmark.RIGHT_SHOULDER,w,h)
        lh,rh = get_xy(lms,mp_pose.PoseLandmark.LEFT_HIP,w,h), \
                get_xy(lms,mp_pose.PoseLandmark.RIGHT_HIP,w,h)
        if None not in (*ls,*rs,*lh,*rh):
            sh_mid  = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
            victim_pt = (int(sh_mid[0] + .33*(hip_mid[0]-sh_mid[0])),
                         int(sh_mid[1] + .33*(hip_mid[1]-sh_mid[1])))
            hand_msg = "Show Rescuer Hands"
        else:
            hand_msg="Show Victim Chest Area"
    else:
        lms=None; hand_msg="Cannot find Victim"

    # ---------- rescuer hand -----------
    if victim_pt and supine_ok and hands_r.multi_hand_landmarks:
        vic_wr = [get_xy(lms,mp_pose.PoseLandmark.LEFT_WRIST,w,h),
                  get_xy(lms,mp_pose.PoseLandmark.RIGHT_WRIST,w,h)] if lms else []
        best_d = 1e9
        for hd in hands_r.multi_hand_landmarks:
            wrist=(int(hd.landmark[0].x*w),int(hd.landmark[0].y*h))
            if any(distance(wrist,vw)<40 for vw in vic_wr if vw[0]): continue
            mp_drawing.draw_landmarks(frame,hd,mp_hands.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())
            for lm in hd.landmark:
                p=(int(lm.x*w),int(lm.y*h)); d=distance(p,victim_pt)
                if d<best_d: best_d, resc_hand = d, p
        if resc_hand:
            cv2.circle(frame,resc_hand,5,(0,255,255),-1)
            if best_d < HAND_PLACEMENT_TOLERANCE:
                hands_centered=True; tgt_color=CHEST_TARGET_COLOR_CENTERED; hand_msg="Hands Centered"
            else: hand_msg="Move Hands Closer to Target"

    # ---------- posture ----------
    if lms and resc_hand:
        lw = get_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST,  w,h)
        rw = get_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, w,h)
        side = "LEFT" if distance(resc_hand,lw) < distance(resc_hand,rw) else "RIGHT"
        ang = elbow_angle(lms, side, w, h)
        if ang:
            posture_ok  = ang >= 160
            posture_msg = "Posture: Good" if posture_ok else f"Posture: Straighten {side.lower()} arm"

    # ---------- readiness & countdown ----------
    ready = hands_centered and posture_ok and supine_ok
    now   = time.time()

    if ready and not cycle_active and not pre_count_active:
        pre_count_active=True; pre_count_start=now

    if pre_count_active and now - pre_count_start >= PRE_START_COUNTDOWN:
        pre_count_active=False
        cycle_active=True
        compressions_left = COMPRESSIONS_PER_SET
        next_beat_time = now          # start metronome

    # ---------- metronome beat ----------
    if cycle_active and next_beat_time is not None:
        if now >= next_beat_time:
            metronome_flash=True
            next_beat_time += BEAT_PERIOD
            if CLICK_WAV: CLICK_WAV.play()
        else:
            metronome_flash=False
    else:
        metronome_flash=False

    # ---------- compression detection / countdown ----------
    if cycle_active and resc_hand:
        y=resc_hand[1]
        if last_hand_y is not None:
            dy = y - last_hand_y
            if dy>COMPRESSION_THRESHOLD and not is_compressing:
                is_compressing=True
            elif dy<-COMPRESSION_THRESHOLD and is_compressing:
                is_compressing=False
                compressions_left -= 1
                if CLICK_WAV is None:   # provide silent tick feedback at compression if no audio
                    metronome_flash = True
                if compressions_left <= 0:
                    cycle_active=False
                    next_beat_time=None
        last_hand_y = y
    else:
        last_hand_y=None; is_compressing=False

    # ---------- UI drawing ----------
    if victim_pt is not None:
        cv2.circle(frame, victim_pt, CHEST_TARGET_RADIUS, tgt_color, -1)
        cv2.circle(frame, victim_pt, CHEST_TARGET_RADIUS+5, tgt_color, 2)

    cv2.putText(frame,f"Placement: {hand_msg}",(10,60),
                cv2.FONT_HERSHEY_DUPLEX,0.8,
                FEEDBACK_COLOR_GOOD if hands_centered else FEEDBACK_COLOR_WARN,2)
    sup_text = SUPINE_STATUS_OK_MSG if supine_ok else SUPINE_STATUS_WARN_MSG
    cv2.putText(frame,sup_text,(10,90),cv2.FONT_HERSHEY_DUPLEX,0.8,
                FEEDBACK_COLOR_GOOD if supine_ok else FEEDBACK_COLOR_ERR,2)
    cv2.putText(frame,f"Rate: {current_rate:.0f} /min",(10,120),
                cv2.FONT_HERSHEY_DUPLEX,0.8,INFO_COLOR,2)
    cv2.putText(frame,posture_msg,(10,150),
                cv2.FONT_HERSHEY_DUPLEX,0.8,
                FEEDBACK_COLOR_GOOD if posture_ok else FEEDBACK_COLOR_WARN,2)
    cv2.putText(frame, TARGET_DEPTH_MSG, (10,180), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, 
                FEEDBACK_COLOR_ERR, 2)

    # pre-start countdown / compressions left
    if pre_count_active:
        cd = int(PRE_START_COUNTDOWN - (now-pre_count_start)) + 1
        cv2.putText(frame,f"Starting in {cd}",(10,210),
                    cv2.FONT_HERSHEY_DUPLEX,1.0,INFO_COLOR,2)
    elif cycle_active:
        cv2.putText(frame,f"Compressions left: {compressions_left}",(10,210),
                    cv2.FONT_HERSHEY_DUPLEX,0.8,INFO_COLOR,2)
    elif ready and not cycle_active:
        cv2.putText(frame,"Cycle complete – reposition to restart",(10,210),
                    cv2.FONT_HERSHEY_DUPLEX,0.7,INFO_COLOR,2)

    # metronome circle
    metro_col = METRONOME_COLOR_BEAT if metronome_flash else METRONOME_COLOR_IDLE
    cv2.circle(frame,(w-50,50),20,metro_col,-1)

    cv2.putText(frame,"TRAINING AID ONLY",(10,h-15),
                cv2.FONT_HERSHEY_DUPLEX,0.6,WARNING_COLOR,2)

    # --- visual guidance -------------------------------------------------
    if victim_pt is not None and resc_hand is not None and not hands_centered:
        cv2.arrowedLine(frame, resc_hand, victim_pt, AR_ARROW_COLOR, 3, tipLength=0.25)

    draw_labels(frame,victim_pt,resc_hand)

    cv2.imshow("CPR Trainer – Proof of Concept",frame)
    if cv2.waitKey(5)&0xFF == ord('q'):
        break

# -------- cleanup --------
pose.close(); hands.close(); cap.release(); cv2.destroyAllWindows()
