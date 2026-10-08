# CV Module — Quick Start

This repository keeps the CV pipeline and RAG retrieval independently runnable.
CV events cross into retrieval only through `data/cv_events/detections.json`.

## Requirements

- Python 3.x
- Git
- CCTV test videos
- YOLO model weights

## 1. Clone

```bash
git clone <YOUR-REPOSITORY-URL>
cd Hacknex-main
```

## 2. Create virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
cd cv
pip install -r requirements.txt
```

## 4. Add YOLO model

Place the required YOLO model inside:

```text
cv/yolo26s.pt
```

Model weights are not included in the Git repository because they are large files.

## 5. Add CCTV videos

Place the test CCTV videos inside:

```text
cv/videos/
```

Example:

```text
cv/videos/
├── cam_01.mp4
├── cam_02.mp4
└── cam_03.mp4
```

## 6. Configure ROI

If an ROI configuration already exists, it will be loaded automatically.

To create a new ROI:

```bash
python set_roi.py --camera cam_03
```

The ROI will be saved under:

```text
cv/roi/
```

## 7. Run the pipeline

```bash
python pipeline.py --camera cam_03 --show
```

For faster testing:

```bash
python pipeline.py --camera cam_03 --show --frame-skip 3 --conf 0.25
```

## 8. Output

Generated results are stored under:

```text
cv/output/
```

including:

```text
cv/output/
├── frames/
├── crops/
├── clips/
└── detections.json
```

The CV pipeline produces:

- object detections
- person tracking
- person-object associations
- temporal confirmation
- evidence scores
- ROI intrusion alerts
- suspicious-object alerts
- timestamps
- camera IDs
- video/clip references
- semantic tags
- searchable event text

## Troubleshooting

### Missing model

If you see a model-not-found error, verify that:

```text
cv/yolo26s.pt
```

exists.

### Missing video

Verify that the requested camera video exists:

```text
cv/videos/cam_03.mp4
```

### No ROI

If no ROI is configured, the pipeline continues without ROI danger alerts.

Create one with:

```bash
python set_roi.py --camera cam_03
```

### Check installation

```bash
python -c "import cv2, numpy, ultralytics; print('CV dependencies OK')"

## CV → RAG integration

Use Python 3.11 for a shared environment, then install both existing dependency
lists without replacing either one:

```powershell
cd D:\Hacknex
py -3.11 -m venv --clear venv
.\venv\Scripts\Activate.ps1
python -m pip install -r cv\requirements.txt -r retrieval\requirements.txt
```

Run the CV pipeline from `cv/`. `--save-json` keeps the normal CV output and
exports the same events to the shared integration file. Evidence paths in the
shared file are project-root-relative; the video remains under `cv/videos/`.

```powershell
cd D:\Hacknex\cv
..\venv\Scripts\python.exe pipeline.py --camera cam_03 --show --frame-skip 2 --save-json
```

Index and query the shared events from the project root. Events are indexed into structured storage without generating embeddings; rerunning indexing upserts by `event_id`.

```powershell
cd D:\Hacknex
.\venv\Scripts\python.exe scripts\index_cv_events.py --input data\cv_events\detections.json
.\venv\Scripts\python.exe scripts\test_cv_rag.py "person carrying a handbag"
.\venv\Scripts\python.exe scripts\test_cv_rag.py "show events from cam_03"
```

The integration check loads and indexes the contract, queries structured
retrieval, prints event and evidence metadata, and checks available
evidence paths on disk.
```
