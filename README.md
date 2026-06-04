# Computer Vision MCI Detection: Eye Blink Rate Analysis

A real-time blink detection system using the Eye Aspect Ratio (EAR) to measure spontaneous blink rate — a candidate non-invasive biomarker for Mild Cognitive Impairment (MCI). Built on MediaPipe's Face Landmarker and OpenCV.

---

## Background: MCI and the Blink Rate Biomarker

**Mild Cognitive Impairment (MCI)** is a clinical stage between normal age-related cognitive decline and dementia. It is characterised by measurable deficits in memory, attention, or executive function that are not yet severe enough to disrupt daily life. Early detection is critical: individuals diagnosed at the MCI stage have a window for intervention before progression to Alzheimer's disease or other dementias.

**Why blink rate?** Spontaneous blink rate (SBR) is regulated in part by the dopaminergic system — specifically, dopamine D2 receptor activity in the basal ganglia and frontal circuits. MCI is associated with early dopaminergic and cholinergic dysfunction in these same circuits. Studies have found that individuals with MCI and prodromal Alzheimer's disease exhibit elevated spontaneous blink rates compared to age-matched healthy controls, suggesting that SBR reflects the underlying neurochemical disruption before gross structural changes become visible on imaging.

Crucially, measuring SBR requires no physical contact, no blood draw, and no specialist equipment — just a camera and a face. This project implements the measurement layer of that pipeline.

---

## The Eye Aspect Ratio (EAR)

### The six landmark points

Each eye is represented by six facial landmarks arranged around the eyelid contour:

```
        p2   p3
        •     •
       / \   / \
      /   \ /   \
p1 •       V       • p4
      \   / \   /
       \ /   \ /
        •     •
        p6   p5
```

| Point | Location |
|-------|----------|
| p1    | Outer corner of the eye |
| p4    | Inner corner (toward the nose) |
| p2    | Upper eyelid, left side |
| p3    | Upper eyelid, right side |
| p5    | Lower eyelid, right side |
| p6    | Lower eyelid, left side |

In MediaPipe's 478-point face mesh, the script maps these as:

- **Left eye:** indices `[362, 385, 387, 263, 373, 380]` → `[p1, p2, p3, p4, p5, p6]`
- **Right eye:** indices `[33, 160, 158, 133, 153, 144]` → `[p1, p2, p3, p4, p5, p6]`

### The formula

EAR is the ratio of the average vertical eyelid opening to the horizontal eye width:

```
        ||p2 - p6|| + ||p3 - p5||
EAR =  ───────────────────────────
               2 × ||p1 - p4||
```

Where `||·||` denotes the Euclidean distance between two points.

**A = ||p2 - p6||** — vertical distance on the left side of the eyelid aperture  
**B = ||p3 - p5||** — vertical distance on the right side of the eyelid aperture  
**C = ||p1 - p4||** — horizontal distance between eye corners (the width)

```
EAR = (A + B) / (2.0 × C)
```

### Worked example: open eye

```
p1 = (100, 150)   p2 = (120, 140)   p3 = (140, 140)
p4 = (160, 150)   p5 = (140, 160)   p6 = (120, 160)

A = ||p2 - p6|| = √[(120-120)² + (140-160)²] = √400 = 20 px
B = ||p3 - p5|| = √[(140-140)² + (140-160)²] = √400 = 20 px
C = ||p1 - p4|| = √[(100-160)² + (150-150)²] = √3600 = 60 px

EAR = (20 + 20) / (2 × 60) = 40 / 120 = 0.333   → eye is OPEN
```

### Worked example: closed eye (blink)

During a blink the eyelids come together — vertical distances collapse while the horizontal width is unchanged:

```
p2 = (120, 149)   p6 = (120, 151)   → A = 2 px  (was 20)
p3 = (140, 149)   p5 = (140, 151)   → B = 2 px  (was 20)
p1 = (100, 150)   p4 = (160, 150)   → C = 60 px (unchanged)

EAR = (2 + 2) / (2 × 60) = 4 / 120 = 0.033   → eye is CLOSED
```

### Why EAR is robust across subjects and distances

The horizontal width C acts as a **normalising denominator**. Because both the vertical distances (A, B) and the horizontal distance (C) are Euclidean lengths measured in the same image pixel space, they scale together when the subject moves closer or further from the camera, or when faces differ in size. The ratio is therefore scale-invariant. It is also partially robust to moderate in-plane rotation because all distances rotate with the face. This means the same threshold applies across different subjects without per-person calibration.

### Why two vertical measurements?

Taking A (left side, p2→p6) and B (right side, p3→p5) independently and averaging them makes the numerator robust to slight asymmetries in the eyelid contour due to squinting, facial expressions, or minor camera angles that compress one side of the eye more than the other.

**Typical ranges:**
- Open eye: EAR 0.25 – 0.40
- Partially closed / drowsy: EAR 0.15 – 0.25
- Fully closed (blink): EAR 0.03 – 0.15

---

## Blink Detection Logic

### Threshold

```python
EAR_THRESHOLD = 0.20
```

A blink is declared when the averaged EAR falls below 0.20. This sits cleanly in the gap between typical open-eye values (≥ 0.25) and closed-eye values (≤ 0.15), giving margin against noise without being so tight that partial blinks are missed.

### Falling-edge detection

Counting every frame where EAR < 0.20 would count a single 150 ms blink as ~4–5 events (at 30 fps). Instead, the script uses a **falling-edge detector** via a boolean flag:

```python
if avg_ear < EAR_THRESHOLD:
    if not eyes_closed:        # transition: open → closed
        blink_count += 1       # count exactly once per blink
        eyes_closed = True
else:
    eyes_closed = False        # reset: eye has reopened
```

A blink is incremented exactly once — at the moment the EAR first drops below threshold. The counter does not increment again until the eye has fully re-opened (EAR ≥ 0.20) and closes again.

### Left/right averaging

```python
avg_ear = (left_ear + right_ear) / 2.0
```

Averaging both eyes before applying the threshold ensures that a unilateral wink does not register as a blink. Voluntary winks will produce roughly half the normal EAR drop, typically staying above 0.20. True bilateral blinks cause both EARs to collapse simultaneously, driving the average well below threshold.

---

## Pipeline: Webcam Frame to Blink Count

```
┌─────────────────────────────────────────────────────────────┐
│  Video source (webcam / file)                               │
│         │                                                   │
│   cap.read() → BGR frame                                    │
│         │                                                   │
│   cvtColor(BGR → RGB)                                       │
│         │                                                   │
│   mp.Image(SRGB, data=rgb_frame)                            │
│         │                                                   │
│   FaceLandmarker.detect_for_video(image, timestamp_ms)      │
│   [MediaPipe Tasks API — VIDEO mode, 478 face landmarks]    │
│         │                                                   │
│   Extract 6 landmarks per eye → pixel coordinates          │
│         │                                                   │
│   eye_aspect_ratio(left_points)  → left_ear                │
│   eye_aspect_ratio(right_points) → right_ear               │
│         │                                                   │
│   avg_ear = (left_ear + right_ear) / 2.0                   │
│         │                                                   │
│   Falling-edge detector                                     │
│   avg_ear < 0.20 AND not eyes_closed → blink_count += 1    │
│         │                                                   │
│   Overlay: blink count, blinks/min, EAR value, landmarks   │
│         │                                                   │
│   cv2.imshow → display frame                                │
└─────────────────────────────────────────────────────────────┘
```

**Timestamp handling:** MediaPipe's Tasks API in VIDEO mode requires monotonically increasing timestamps in milliseconds. The script derives these from `frame_count / fps`, giving a synthetic but strictly increasing timeline regardless of actual wall-clock processing speed. This ensures the landmarker's internal temporal smoothing operates correctly.

**Model:** `face_landmarker.task` (MediaPipe's float16 face landmarker). The script auto-downloads this file on first run if it is not present. The file is gitignored — it is a large binary not suited for version control.

---

## Roadmap

This project implements the **measurement layer** of a broader MCI detection pipeline. Planned extensions:

| Layer | Description |
|-------|-------------|
| **Gaze tracking** | Estimate gaze direction from iris landmark centroid relative to eye corners. MCI and early Alzheimer's are associated with impaired smooth pursuit and increased saccadic intrusions. |
| **Head pose estimation** | Derive pitch/yaw/roll from the 3D face mesh. Head pose is required to correctly interpret gaze vectors and to filter frames where the face is occluded or angled beyond reliable landmark detection. |
| **Temporal analysis** | Aggregate blink rate over rolling windows (30 s, 5 min) rather than session totals. Burst patterns and inter-blink intervals carry additional diagnostic signal. |
| **Logging + export** | Write per-session CSV with frame-level EAR, blink timestamps, and derived blink rate for downstream statistical analysis. |

---

## Installation

```bash
pip install opencv-python mediapipe numpy
```

Python 3.8+ required. Tested on Python 3.11.

The MediaPipe face landmarker model (`face_landmarker.task`, ~22 MB) is downloaded automatically on first run.

---

## Usage

```bash
python blink_recognition.py
```

You will be prompted to choose a video source:

```
Blink Detection System
Choose video source:
1. Webcam (live)
2. Video file
Enter choice (1 or 2):
```

**Webcam mode (option 1):** Opens the default camera (index 0). The frame is mirrored for a natural viewing experience. Press `q` to quit.

**Video file mode (option 2):** Enter the path to an `.mp4`, `.avi`, or similar file. Playback is paced to the file's native FPS. Processing stops automatically at the end of the video.

On exit, the terminal prints:

```
FINAL STATISTICS
Total Blinks: 14
Video Duration: 60.00 seconds
Processing Time: 61.23 seconds
Average Blinks/Minute: 14.00
Frames Processed: 1800
```

**On-screen overlay:**
- `Blinks:` — cumulative blink count
- `Blinks/Min:` — rolling blinks per minute based on elapsed video time
- `EAR:` — current averaged eye aspect ratio
- Green dots — the 12 tracked eyelid landmarks (6 per eye)
- `BLINK!` — flashes red when a blink is detected
