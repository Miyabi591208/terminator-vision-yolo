# Terminator Vision (YOLO HUD Demo)

映画『ターミネーター』の視点風HUDを、物体検出AI（YOLO）で再現したデモです。  
入力（動画/画像）に対して YOLO で物体検出を行い、検出結果をターミネーター風UIとして重ね描画します。

> This project is for **visual effect / educational demonstration** purposes only.

## Demo
- Output sample: `outputs/out_terminator_sample.mp4`
- (Large original videos are stored via Git LFS / or external link)

## Features
- YOLO object detection (Ultralytics)
- Terminator-style HUD overlay (bbox / confidence / target lock)
- Optional tracking IDs (when using track mode)

## Quickstart (Google Colab)
1. Open `terminator_vision.ipynb` in Colab
2. Run all cells
3. Upload your video and generate `out_terminator.mp4`

## Local Run
```bash
pip install -r requirements.txt
python src/terminator_vision.py --source path/to/video.mp4 --target person --out out_terminator.mp4
