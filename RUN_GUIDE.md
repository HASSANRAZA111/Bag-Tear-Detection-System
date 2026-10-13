# Run Guide

The app uses one model file you train yourself: `models/best.pt`. (A hosted API cannot name damage types, so there is none.)

---

## 1. Install (once)

Python 3.10 or newer.

```bash
cd bag-tear-detector
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the datasets

Download each as **YOLOv8** format zip (Roboflow: open page, *Download Dataset*, *YOLOv8*, *download zip*; free login needed). Keep the zips as they are.

| Dataset | Gives |
|---|---|
| [2-BagDamageHandle](https://universe.roboflow.com/project1bag-and-handle-detection/2-bagdamagehandle) | **bag**, tear (real bags) |
| Damage detection dataset (class `Damage-bag`) | tear (real bags) |
| [Defect-Clothes](https://universe.roboflow.com/hawadz/defect-clothes-ukrks) | tear, hole, stain |
| [Fabric Defect Detection](https://universe.roboflow.com/sqc/fabric-defect-detection-ehhyx) | tear, hole, stain |
| [Textile Damage Identifier](https://universe.roboflow.com/hidayatul-husna/textile-damage-identifier) | few of each |
| [Leather defect](https://universe.roboflow.com/myself-n7rmq/leather-defect-detction) | hole, tear |
| [Broken stitch detection](https://universe.roboflow.com/brack/broken-stitch-detection) | stitching |
| [scratch](https://universe.roboflow.com/monkey-pafcf/scratch-g4gtg) | scratch (phone screens) |

Only the first two are real bags, so expect weaker results on the other types. Your own labelled bag photos help most (see section 7).

## 3. Train on Google Colab (free GPU)

1. Zip the project folder **without** the `data/` folder.
2. Open [colab.research.google.com](https://colab.research.google.com), *File > Upload notebook*, choose `training/colab_train.ipynb`.
3. *Runtime > Change runtime type > T4 GPU*.
4. Run the cells top to bottom. Upload the project zip, then **all dataset zips** when asked.
5. Check the merge output: `Ready: data/bag_tear/data.yaml`. `bag` should be about 2,400 and `tear` above 5,000.
6. Training takes about 1.5 hours. At the end `best.pt` downloads.
7. Put it at `bag-tear-detector/models/best.pt`.

For better accuracy, change the training cell to:
```
!python training/train.py --data data/bag_tear/data.yaml --model yolo11s.pt --epochs 100 --imgsz 800
```

## 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. Upload photos, choose the view, click Analyze. If damage is missed, lower the **confidence** slider to about 0.15.

## 5. Deploy free (Streamlit Community Cloud)

```bash
git add -f models/best.pt
git add .
git commit -m "update model"
git push
```

Then at [share.streamlit.io](https://share.streamlit.io) choose *Create app*, pick your repo, branch `main`, file `app.py`. Later updates redeploy on every push. No secrets are needed.

If the app crashes with an out-of-memory error, reboot it. Free hosting has about 1 GB of memory.

## 6. Make it better with your own photos

1. Take 100+ photos per damage type on real bags.
2. Label them with [Roboflow](https://roboflow.com), [CVAT](https://app.cvat.ai) or [Label Studio](https://labelstud.io).
3. Use exactly these class names: `tear`, `scratch`, `stain`, `stitching`, `hole`, `bag`. Box the whole bag (`bag`) in every photo.
4. Export as YOLOv8 and upload that zip with the others in step 3.

## Credits

Datasets: Roboflow Universe (mostly CC BY 4.0, credit the authors; check each licence). Model: [Ultralytics YOLO11](https://docs.ultralytics.com). App: [Streamlit](https://streamlit.io).