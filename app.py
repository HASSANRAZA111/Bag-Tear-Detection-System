"""Bag Tear Detector - Streamlit app.

Run:  streamlit run app.py
"""
from datetime import datetime

import streamlit as st
from PIL import Image

from bagtear.detectors import DetectorUnavailable, build_detector
from bagtear.pipeline import ImageInput, analyze, guess_view
from bagtear.progress import steps_markdown
from bagtear.labels import DAMAGE_TYPES
from bagtear.schema import VIEWS
from bagtear.settings import Settings

st.set_page_config(page_title="Bag Tear Detector", page_icon="👜", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_detector(model_path: str, modified: float):
    """`modified` is only part of the cache key: replacing best.pt reloads the model."""
    return build_detector(Settings.from_env({"BAGTEAR_MODEL": model_path}))


settings = Settings.from_env()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05,
                     help="Lower = finds more damage but more false alarms.")
    show_context = st.checkbox("Show bag / handle boxes", value=False,
                               help="Helps check how the position was worked out.")
    st.caption("Positions: left/right as seen in the photo. "
               "'Seam' = close to the bag's outer edge.")

st.title("👜 Bag Tear Detector")
st.write("Upload one or more photos of **one bag** (front, back, side). "
         "The app finds damage (tear, scratch, stain, stitching, hole), says where it is and "
         "how severe it is, and gives a short report.")

# ------------------------------------------------------------------ detector
try:
    mtime = settings.model_path.stat().st_mtime if settings.model_path.exists() else 0.0
    detector = load_detector(str(settings.model_path), mtime)
except DetectorUnavailable as e:
    st.error("Model not available")
    st.code(str(e), language=None)
    st.stop()
except Exception as e:  # corrupt weights, missing libraries, ...
    st.error(f"Could not load the model: {e}")
    st.stop()

known = getattr(detector, "damage_types", None)
if known is not None:
    st.caption(f"Model: **{settings.model_path.name}** · damage types it can detect: "
               f"**{', '.join(known) if known else 'none found'}**")
    missing = [t for t in DAMAGE_TYPES if t not in known]
    if missing:
        st.info(f"This model cannot detect: {', '.join(missing)}. A model only reports the "
                "types it was trained on. To add them, train with data that labels them "
                "(RUN_GUIDE.md, Part C).")

# ------------------------------------------------------------------ upload
files = st.file_uploader("Bag photos", type=["jpg", "jpeg", "png", "webp"],
                         accept_multiple_files=True)

inputs = []
if files:
    st.subheader("Which side is each photo?")
    cols = st.columns(min(len(files), 4))
    for i, f in enumerate(files):
        with cols[i % len(cols)]:
            try:
                img = Image.open(f)
                img.load()
            except Exception:
                st.warning(f"{f.name}: not a readable image, skipped.")
                continue
            st.image(img, width="stretch")
            guess = guess_view(f.name)
            view = st.selectbox(f.name, VIEWS, index=VIEWS.index(guess),
                                key=f"view_{i}_{f.name}")
            inputs.append(ImageInput(f.name, view, img))

if st.button("Analyze", type="primary", disabled=not inputs):
    views = [i.view for i in inputs]
    n = len(inputs)
    with st.status("Analyzing photos...", expanded=True) as status:
        board = st.empty()
        board.markdown(steps_markdown(n, views, "detect", 0, n))

        def on_progress(stage, done, total):
            board.markdown(steps_markdown(n, views, stage, done, total))

        try:
            st.session_state["result"] = analyze(detector, inputs, conf=conf,
                                                 show_context=show_context,
                                                 on_progress=on_progress)
            st.session_state["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            board.markdown(steps_markdown(n, views, "done"))
            status.update(label="Inspection complete", state="complete", expanded=True)
        except Exception as e:
            st.session_state.pop("result", None)
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis failed: {e}")

# ------------------------------------------------------------------ results
result = st.session_state.get("result")
if result:
    rep = result.report
    st.divider()

    # Overall status card: result, counts, severity and damage types
    if rep.total:
        st.error(f"### 🔴 Damage detected — inspection required\n\n{rep.stats_line()}\n\n"
                 f"**{rep.severity_line()}**\n\n{rep.types_line()}")
    else:
        st.success(f"### 🟢 No damage detected\n\n{rep.stats_line()}\n\n"
                   f"**{rep.severity_line()}**")

    st.subheader("Summary")
    (st.warning if rep.total else st.success)(rep.summary)

    st.subheader("Annotated images")
    cols = st.columns(min(len(result.annotated), 3))
    for i, a in enumerate(result.annotated):
        with cols[i % len(cols)]:
            tag = f"{a.n_tears} damage area(s)" if a.n_tears else "no damage detected"
            st.image(a.image, caption=f"{a.view.capitalize()} - {a.name} ({tag})",
                     width="stretch")

    st.subheader("Damage locations")
    st.dataframe(rep.table_rows(), width="stretch", hide_index=True)

    st.subheader("Download report")
    st.markdown("Save the inspection report (result, summary and tear locations) "
                "as a text (.txt) file.")
    st.download_button("Download report",
                       rep.to_text(generated=st.session_state.get("analyzed_at")),
                       "bag_inspection_report.txt", "text/plain")
