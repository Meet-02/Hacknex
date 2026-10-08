# CV Module — Quick Start

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
```