# Bag Tear Detector

Upload photos of a bag (front, back, side). The app finds the damage, names its **type** (tear, scratch, stain, stitching, hole), says **where on the bag** it is, rates the **severity**, and gives you annotated images plus a plain-language report.

> Example: "3 damage areas found on this bag (2 tears, 1 hole). Overall severity: Moderate. The back panel has three damage areas: tear, bottom-left, near the seam (Minor); ..."

## Features

- One or more photos per bag, each labelled front / back / side
- Damage types: **tear, scratch, stain, stitching, hole**, drawn as colour-coded numbered boxes
- Location on the bag, for example "back panel, bottom-left, near the seam, near the handle"
- Severity per damage and overall: **Minor** (under 1% of the bag), **Moderate** (1% to under 5%), **Severe** (5% or more)
- Status card (red or green), summary, damage table, downloadable `.txt` report
- Runs on your own YOLO model (`models/best.pt`), free to host

## How it works

```
photos -> YOLO11 model (tear, scratch, stain, stitching, hole, bag)
       -> position of each damage inside the bag box (3x3 grid, edge band = "seam")
       -> severity = damage box area / bag box area
       -> one combined report per bag
```

The location and severity logic is plain geometry (`bagtear/`), not machine learning.

## Quick start

```bash
git clone <your-repo-url>
cd bag-tear-detector
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app needs a trained model at `models/best.pt`. See **[RUN_GUIDE.md](RUN_GUIDE.md)** to train one for free on Google Colab.

## Project layout

```
app.py                  Streamlit web app
bagtear/                detection, location, severity, report code
training/               dataset merge script, train script, Colab notebook
configs/class_map.yaml  maps dataset class names to our types
data/raw/               put downloaded datasets here (not committed)
models/best.pt          your trained model
tests/                  automated tests
RUN_GUIDE.md            step-by-step guide
```

## Current model results

Latest training run (YOLO11n, 80 epochs, test set the model never saw):

| Type | mAP50 | Note |
|---|---|---|
| bag | 0.90 | good |
| scratch | 0.98 | trained on phone screens, so not reliable on bags |
| tear | 0.37 | finds about 3 in 10 |
| stain | 0.57 | fabric data |
| hole | 0.28 | weak |
| stitching | n/a | only 93 training boxes, test set too small to measure |

## Limitations

- **Accuracy is limited by the data.** Only the bag datasets are real bags. Stain, hole and stitching come from fabric, clothes and leather. Scratch comes from phone screens. Adding your own labelled bag photos is the best way to improve it.
- A model can only name a type it was trained on. A type with no training boxes is never detected.
- Severity is box area / bag area. Thin diagonal damage has a big box, so severity can be overestimated.
- Each photo is analysed separately, so damage visible in two photos can be counted twice.
- If the bag outline is not found in a photo, position and severity use the whole photo instead.
- Not verified: accuracy on your real bags and long-term hosting limits. The free Streamlit tier has about 1 GB of memory.

## Datasets and credits

Datasets come from [Roboflow Universe](https://universe.roboflow.com) (mostly CC BY 4.0, so credit the authors) and similar sources. The full list with links is in [RUN_GUIDE.md](RUN_GUIDE.md). Check each dataset's licence before commercial use.

Built with [Ultralytics YOLO11](https://docs.ultralytics.com) (AGPL-3.0) and [Streamlit](https://streamlit.io).