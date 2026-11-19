"""Streamlit interface for controlled-background multi-item waste analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from model_config import CHECKPOINT_PATH, CLASS_NAMES, QUALITY_THRESHOLD
from predictor import PredictionError, WastePredictor
from scene_analysis import (
    SceneAnalysisError,
    SceneAnalysisResult,
    analyze_scene,
    annotated_png_bytes,
    item_csv_bytes,
    load_uploaded_image,
)


st.set_page_config(
    page_title="Waste Scene Sorting Prototype",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8f7; color: #142421; }
    .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero {
        background: linear-gradient(135deg, #0f5f59 0%, #173f50 100%);
        color: white; border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;
        box-shadow: 0 14px 36px rgba(20, 58, 62, 0.16);
    }
    .hero-label { color: #a7f3d0; font-size: .78rem; font-weight: 800; letter-spacing: .12em; }
    .hero h1 { color: white; font-size: 2.45rem; line-height: 1.08; margin: .35rem 0 .55rem; }
    .hero p { color: #e7f7f3; font-size: 1.03rem; max-width: 850px; margin: 0; }
    .panel {
        background: white; border: 1px solid #dfe8e5; border-radius: 18px;
        padding: 1.1rem 1.25rem; margin-bottom: 1rem;
    }
    .eyebrow { color: #0f766e; font-size: .78rem; font-weight: 800; letter-spacing: .09em; }
    .manual { color: #9a3412; font-weight: 800; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #dfe8e5; padding: .8rem 1rem;
        border-radius: 14px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 12px; min-height: 2.8rem; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local ResNet-18 checkpoint…")
def load_predictor(checkpoint: str) -> WastePredictor:
    """Load the checkpoint once across ordinary Streamlit reruns."""
    return WastePredictor(Path(checkpoint))


def result_key(image_bytes: bytes, selected_bin: str | None) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{digest}:{selected_bin or 'no-target'}"


def item_rows(result: SceneAnalysisResult) -> list[dict[str, object]]:
    return [
        {
            "Item": item.item_number,
            "Predicted class": item.predicted_class,
            "Top score": f"{item.top_score:.1%}",
            "Suggested category": item.suggested_category,
            "Review": "Review" if item.review_required else "Ready",
            "Target-bin check": item.quality_control_result,
            "Box (x1, y1, x2, y2)": (
                f"{item.box.x_min}, {item.box.y_min}, "
                f"{item.box.x_max}, {item.box.y_max}"
            ),
        }
        for item in result.items
    ]


st.markdown(
    """
    <section class="hero">
      <div class="hero-label">PROJECT 07 · LOCAL TWO-STAGE PROTOTYPE</div>
      <h1>Multi-item waste sorting on a controlled background</h1>
      <p>Upload one image containing separated items on a plain, contrasting background.
      The app first finds foreground regions with classical image processing, then applies
      the saved six-class ResNet-18 to each crop.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Demonstration guide")
    st.markdown(
        "1. Use **2–4 separated objects**.\n"
        "2. Choose a **plain, contrasting background**.\n"
        "3. Leave space between objects.\n"
        "4. Click **Analyze scene**."
    )
    st.divider()
    st.markdown("### What the stages mean")
    st.caption(
        "Stage 1 is an explainable OpenCV foreground mask and connected-region "
        "localizer. It is not YOLO and is not a learned detector."
    )
    st.caption(
        "Stage 2 classifies each crop as cardboard, glass, metal, paper, plastic, "
        "or trash with the saved ResNet-18 checkpoint."
    )
    st.warning(
        "The 90.98% held-out accuracy belongs to single-item TrashNet "
        "classification—not multi-object localization or contamination detection."
    )

st.markdown('<div class="eyebrow">1 · INPUT</div>', unsafe_allow_html=True)
input_left, input_right = st.columns([1.65, 1], gap="large")
with input_left:
    uploaded_file = st.file_uploader(
        "Scene image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Maximum 20 MB and 20 million pixels.",
    )
with input_right:
    use_target_bin = st.checkbox(
        "Enable target-bin check",
        help="Optional. Sorting suggestions work without a target bin.",
    )
    selected_bin = st.selectbox(
        "Target bin",
        CLASS_NAMES,
        disabled=not use_target_bin,
    )
    active_target = selected_bin if use_target_bin else None

analyze_clicked = st.button(
    "Analyze scene",
    type="primary",
    disabled=uploaded_file is None,
    width="stretch",
)

image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
current_key = result_key(image_bytes, active_target) if image_bytes else None

if analyze_clicked and image_bytes is not None:
    try:
        image = load_uploaded_image(image_bytes)
        predictor = load_predictor(str(CHECKPOINT_PATH))
        with st.spinner("Localizing separated regions and classifying their crops…"):
            analysis = analyze_scene(image, predictor, active_target)
    except (SceneAnalysisError, PredictionError) as error:
        st.session_state.pop("scene_analysis", None)
        st.session_state.pop("scene_analysis_key", None)
        st.error(str(error))
    else:
        st.session_state.scene_analysis = analysis
        st.session_state.scene_analysis_key = current_key
        st.session_state.scene_upload_name = uploaded_file.name

stored_result = st.session_state.get("scene_analysis")
stored_key = st.session_state.get("scene_analysis_key")
if stored_result is not None and stored_key != current_key:
    st.info("The image or target-bin option changed. Click **Analyze scene** to refresh results.")
    stored_result = None

if isinstance(stored_result, SceneAnalysisResult):
    result = stored_result
    st.markdown('<div class="eyebrow">2 · LOCALIZED SCENE</div>', unsafe_allow_html=True)
    if not result.items:
        st.image(result.original_image, caption="Uploaded image", width="stretch")
        st.warning(result.localization.message)
    else:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Localized items", len(result.items))
        metric_columns[1].metric("Review flags", result.review_count)
        metric_columns[2].metric(
            "Foreground threshold",
            f"{result.localization.color_distance_threshold:.0f}",
        )
        metric_columns[3].metric(
            "Target bin", result.selected_bin or "None"
        )

        original_column, annotated_column = st.columns(2, gap="large")
        with original_column:
            st.image(
                result.original_image,
                caption="Original upload",
                width="stretch",
            )
        with annotated_column:
            st.image(
                result.annotated_image,
                caption="Numbered localized regions and crop predictions",
                width="stretch",
            )

        if st.checkbox(
            "Show foreground mask",
            help=(
                "White pixels passed the background-contrast rule. Disconnected "
                "fragments may be reconciled into one object box afterward."
            ),
        ):
            st.image(
                result.localization.foreground_mask,
                caption=(
                    "Stage 1 foreground mask — white is foreground, black is "
                    "estimated background"
                ),
                clamp=True,
                width="stretch",
            )

        st.markdown('<div class="eyebrow">3 · SORTING SUMMARY</div>', unsafe_allow_html=True)
        count_columns = st.columns(max(1, len(result.counts_by_suggestion)))
        for column, (category, count) in zip(
            count_columns, sorted(result.counts_by_suggestion.items())
        ):
            column.metric(category.title(), count)

        st.dataframe(item_rows(result), hide_index=True, width="stretch")

        for item in result.items:
            review_label = " · MANUAL REVIEW" if item.review_required else ""
            with st.expander(
                f"Item {item.item_number}: {item.predicted_class} "
                f"({item.top_score:.1%}){review_label}"
            ):
                st.write(f"**Suggested category:** {item.suggested_category}")
                st.write(f"**Quality check:** {item.quality_control_result}")
                score_rows = [
                    {"Class": class_name, "Softmax score": item.scores[class_name]}
                    for class_name in CLASS_NAMES
                ]
                st.dataframe(
                    score_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Softmax score": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.4f"
                        )
                    },
                )

        download_left, download_right = st.columns(2)
        base_name = Path(st.session_state.get("scene_upload_name", "scene")).stem
        download_left.download_button(
            "Download annotated PNG",
            data=annotated_png_bytes(result),
            file_name=f"{base_name}_annotated.png",
            mime="image/png",
            width="stretch",
        )
        download_right.download_button(
            "Download per-item CSV",
            data=item_csv_bytes(result),
            file_name=f"{base_name}_items.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(result.localization.message)

st.divider()
st.markdown(
    f"**Interpret results cautiously.** Scores below {QUALITY_THRESHOLD:.2f} are sent "
    "to manual review. Softmax scores are not calibrated guarantees. Boxes are "
    "foreground regions from a controlled-background heuristic, not learned object "
    "detections. This app does not measure purity, multi-object precision, mAP, IoU, "
    "latency, FPS, or conveyor performance."
)
