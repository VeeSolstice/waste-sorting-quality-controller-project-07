"""Streamlit interface for the default workflow and optional learned detector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from app_mode_state import (
    APP_MODES,
    CONTROLLED_MODE,
    EXPERIMENTAL_MODE,
    isolate_mode_state,
)
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
from taco_detector_inference import (
    DISPLAY_SCORE_THRESHOLD,
    SELECTION_RECORD_PATH,
    DetectorAnalysisResult,
    DetectorInputError,
    DetectorSetupError,
    LoadedTacoDetector,
    analyze_uploaded_image,
    load_selected_detector,
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
    .eyebrow { color: #0f766e; font-size: .78rem; font-weight: 800; letter-spacing: .09em; }
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
    """Load the classification checkpoint once across ordinary reruns."""
    return WastePredictor(Path(checkpoint))


@st.cache_resource(show_spinner="Verifying and loading the selected TACO detector…")
def load_experimental_detector(selection_record: str) -> LoadedTacoDetector:
    """Verify and cache only the detector named in the selection record."""
    return load_selected_detector(Path(selection_record))


def result_key(image_bytes: bytes, selected_bin: str | None = None) -> str:
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


def render_controlled_mode() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-label">PROJECT 07 · PRIMARY CONTROLLED WORKFLOW</div>
          <h1>Multi-item waste sorting on a controlled background</h1>
          <p>Upload separated items on a plain, contrasting background. OpenCV finds
          foreground regions and the saved TrashNet ResNet-18 classifies each crop.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### Controlled demonstration guide")
        st.markdown(
            "1. Use **2–4 separated objects**.\n"
            "2. Choose a **plain, contrasting background**.\n"
            "3. Leave space between objects.\n"
            "4. Click **Analyze controlled scene**."
        )
        st.caption(
            "Stage 1 is an explainable OpenCV foreground localizer. Stage 2 uses "
            "the saved six-class TrashNet ResNet-18 on each crop."
        )
        st.warning(
            "The 90.98% held-out accuracy is single-item TrashNet classification, "
            "not multi-object localization or contamination accuracy."
        )

    st.markdown('<div class="eyebrow">1 · INPUT</div>', unsafe_allow_html=True)
    input_left, input_right = st.columns([1.65, 1], gap="large")
    with input_left:
        uploaded_file = st.file_uploader(
            "Controlled-background scene image",
            type=("jpg", "jpeg", "png", "bmp", "webp"),
            help="Maximum 20 MB and 20 million pixels.",
            key="controlled_upload",
        )
    with input_right:
        use_target_bin = st.checkbox(
            "Enable target-bin check",
            help="Optional. Sorting suggestions work without a target bin.",
        )
        selected_bin = st.selectbox(
            "Target bin", CLASS_NAMES, disabled=not use_target_bin
        )
        active_target = selected_bin if use_target_bin else None

    analyze_clicked = st.button(
        "Analyze controlled scene",
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
            with st.spinner("Localizing regions and classifying crops…"):
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
    if stored_result is not None and st.session_state.get("scene_analysis_key") != current_key:
        st.info("The image or target-bin option changed. Analyze again to refresh results.")
        stored_result = None

    if isinstance(stored_result, SceneAnalysisResult):
        result = stored_result
        st.markdown(
            '<div class="eyebrow">2 · LOCALIZED SCENE</div>',
            unsafe_allow_html=True,
        )
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
            metric_columns[3].metric("Target bin", result.selected_bin or "None")

            original_column, annotated_column = st.columns(2, gap="large")
            with original_column:
                st.image(result.original_image, caption="Original upload", width="stretch")
            with annotated_column:
                st.image(
                    result.annotated_image,
                    caption="OpenCV regions and ResNet crop predictions",
                    width="stretch",
                )

            if st.checkbox("Show foreground mask"):
                st.image(
                    result.localization.foreground_mask,
                    caption="OpenCV foreground mask — white is estimated foreground",
                    clamp=True,
                    width="stretch",
                )

            st.markdown(
                '<div class="eyebrow">3 · SORTING SUMMARY</div>',
                unsafe_allow_html=True,
            )
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
                        {"Class": name, "Softmax score": item.scores[name]}
                        for name in CLASS_NAMES
                    ]
                    st.dataframe(score_rows, hide_index=True, width="stretch")

            base_name = Path(
                st.session_state.get("scene_upload_name", "scene")
            ).stem
            download_left, download_right = st.columns(2)
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
        f"**Interpret results cautiously.** Scores below {QUALITY_THRESHOLD:.2f} are "
        "sent to manual review. Softmax scores are not calibrated guarantees. "
        "Target-bin checks are prediction-based rules, not measured purity or "
        "contamination decisions."
    )


def _experimental_setup_message() -> str | None:
    if not SELECTION_RECORD_PATH.is_file():
        return (
            "The ignored local TACO selection record is missing. Run the documented "
            "detector selection command and restore its selected checkpoint under data/."
        )
    try:
        selection = json.loads(SELECTION_RECORD_PATH.read_text(encoding="utf-8"))
        checkpoint_path = Path(selection["selected_checkpoint_path"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return "The local TACO selection record is unreadable or incomplete."
    if not checkpoint_path.is_file():
        return (
            "The selected ignored TACO checkpoint is missing. Restore the exact local "
            "checkpoint named by the selection record; the app will not download one."
        )
    return None


def render_experimental_mode() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-label">OPTIONAL · EXPERIMENTAL LEARNED DETECTOR</div>
          <h1>Experimental TACO litter-scene detection</h1>
          <p>This optional mode draws learned boxes for outdoor litter scenes. It is
          separate from the primary controlled-background TrashNet workflow.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### Experimental mode")
        st.warning(
            "No target-bin pass/fail, purity, or contamination decision is available "
            "in this mode."
        )
        st.caption(
            "Held-out TACO results: box AP50 0.1846, AP50:95 0.1049, glass AP50 0. "
            "These are outdoor TACO litter results—not tabletop or conveyor results."
        )

    setup_message = _experimental_setup_message()
    if setup_message:
        st.error(setup_message)
        st.info(
            "The default controlled-background mode remains available and does not "
            "depend on the TACO checkpoint."
        )

    st.markdown('<div class="eyebrow">1 · LITTER IMAGE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Outdoor litter image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Maximum 20 MB, 20 million pixels, and 6000 pixels per side.",
        key="experimental_upload",
    )
    analyze_clicked = st.button(
        "Run experimental detector",
        type="primary",
        disabled=uploaded_file is None or setup_message is not None,
        width="stretch",
    )
    image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    current_key = result_key(image_bytes) if image_bytes else None

    if analyze_clicked and image_bytes is not None:
        try:
            detector = load_experimental_detector(str(SELECTION_RECORD_PATH))
            with st.spinner("Running the validation-selected learned detector…"):
                analysis = analyze_uploaded_image(
                    image_bytes, detector, DISPLAY_SCORE_THRESHOLD
                )
        except (DetectorSetupError, DetectorInputError, OSError, RuntimeError) as error:
            st.session_state.pop("detector_analysis", None)
            st.session_state.pop("detector_analysis_key", None)
            st.error(str(error))
        else:
            st.session_state.detector_analysis = analysis
            st.session_state.detector_analysis_key = current_key
            st.session_state.detector_upload_name = uploaded_file.name

    stored_result = st.session_state.get("detector_analysis")
    if stored_result is not None and st.session_state.get("detector_analysis_key") != current_key:
        st.info("The uploaded litter image changed. Run the detector again to refresh.")
        stored_result = None

    if isinstance(stored_result, DetectorAnalysisResult):
        result = stored_result
        st.markdown(
            '<div class="eyebrow">2 · EXPERIMENTAL PREDICTIONS</div>',
            unsafe_allow_html=True,
        )
        original_column, annotated_column = st.columns(2, gap="large")
        with original_column:
            st.image(
                result.original_image,
                caption="Orientation-corrected upload",
                width="stretch",
            )
        with annotated_column:
            st.image(
                result.annotated_image,
                caption="Learned TACO boxes above the display cutoff",
                width="stretch",
            )

        if not result.detections:
            st.warning(
                "No prediction cleared the display cutoff. Manual review is needed."
            )
        else:
            rows = [
                {
                    "Predicted class": detection.display_name,
                    "Model score": detection.score,
                    "Box (x1, y1, x2, y2)": ", ".join(
                        f"{value:.1f}" for value in detection.box_xyxy
                    ),
                }
                for detection in result.detections
            ]
            st.metric("Displayed detections", len(result.detections))
            st.dataframe(rows, hide_index=True, width="stretch")

        st.caption(
            f"Display-only cutoff: {result.display_score_threshold:.2f}. The installed "
            f"Torchvision ROI head first applies its internal "
            f"{result.model_internal_score_threshold:.2f} threshold and keeps at most "
            f"{result.maximum_detections_per_image} detections. OTHER is displayed as "
            "‘annotated waste needing review’; it is not the TrashNet trash class."
        )

    st.divider()
    st.warning(
        "Experimental predictions only. Held-out TACO box AP50 is 0.1846, AP50:95 "
        "is 0.1049, and glass AP50 is 0. This mode does not determine bin matching, "
        "contamination, purity, or factory-conveyor performance."
    )


with st.sidebar:
    st.markdown("## Analysis mode")
    active_mode = st.radio(
        "Choose workflow",
        APP_MODES,
        index=0,
        help="The controlled-background workflow remains the primary default.",
    )
    st.divider()

if isolate_mode_state(st.session_state, active_mode):
    st.toast("Mode changed; results from the other workflow were cleared.")

if active_mode == CONTROLLED_MODE:
    render_controlled_mode()
elif active_mode == EXPERIMENTAL_MODE:
    render_experimental_mode()
