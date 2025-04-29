# CPR-Trainer

**CPR-Trainer** is a computer-vision training aid that uses your webcam, OpenCV, and MediaPipe to give real-time feedback while you practice chest compressions and rescue breaths on a mannequin (or a consenting volunteer!).  
On-screen graphics guide hand placement, compression depth & rate, elbow posture, victim position, and the **30 compressions + 2 breaths** cycle—all timed to a built-in metronome.

> **Training purposes only** – this project does *not* replace formal CPR certification.

---

## Features

- **Pose & hand tracking** – MediaPipe detects the victim’s posture and the rescuer’s hand placement and elbow angle.  
- **Hand-placement target** – Bull’s-eye overlay plus arrow to guide your hands.  
- **Metronome** – Visual flash and optional `click.wav` at **110 BPM** keeps rhythm.  
- **Compression detection** – Counts compressions from wrist motion.  
- **Feedback system** – Live text prompts and color-coded signals (green = good, yellow = warn, red = error).  
- **Breath phase instructions** – Shows step-by-step prompts for giving 2 rescue breaths after each set.  
- **Full training cycle** – Completes **4 sets** of 30 compressions + 2 breaths.  
- **Audio integration** – Optional click sound if `pygame` is available.  

---

## Quick start

```bash
# 1 clone
git clone https://github.com/<your-username>/CPR-Trainer.git
cd CPR-Trainer

# 2 (optional) virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3 install dependencies
pip install -r requirements.txt
# or individually:
# pip install opencv-python mediapipe pygame numpy

# 4 (optional) metronome sound
cp extras/click.wav .

# 5 run
python cpr_trainer.py