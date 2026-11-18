"""Simple Tkinter interface for single-item waste image analysis."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, UnidentifiedImageError

from model_config import CHECKPOINT_PATH, CLASS_NAMES, SOFTMAX_NOTICE
from predictor import PredictionError, WastePredictor


class WasteClassifierApp:
    """A small desktop interface that loads its checkpoint only once."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Waste Sorting Quality Controller")
        self.root.minsize(650, 610)
        self.selected_image: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.predictor: WastePredictor | None = None

        self.image_path_text = tk.StringVar(value="No image selected")
        self.target_bin = tk.StringVar(value=CLASS_NAMES[0])
        self.predicted_class = tk.StringVar(value="—")
        self.top_score = tk.StringVar(value="—")
        self.quality_result = tk.StringVar(value="—")
        self.checkpoint_status = tk.StringVar(value="Loading checkpoint…")

        self._build_interface()
        self._load_checkpoint_once()

    def _build_interface(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame,
            text="Waste Sorting Quality Controller",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(0, 6))
        ttk.Label(
            frame,
            text="Prototype input: one main waste item per image.",
        ).grid(row=1, column=0, columnspan=2, pady=(0, 12))

        ttk.Button(frame, text="Choose image…", command=self.choose_image).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(frame, textvariable=self.image_path_text, wraplength=470).grid(
            row=2, column=1, sticky="w", padx=(12, 0)
        )

        self.preview_label = ttk.Label(
            frame, text="Image preview", anchor="center", relief="solid"
        )
        self.preview_label.grid(
            row=3, column=0, columnspan=2, sticky="nsew", pady=12, ipady=90
        )

        ttk.Label(frame, text="Target bin:").grid(row=4, column=0, sticky="w")
        target_selector = ttk.Combobox(
            frame,
            textvariable=self.target_bin,
            values=CLASS_NAMES,
            state="readonly",
            width=22,
        )
        target_selector.grid(row=4, column=1, sticky="w")

        self.analyze_button = ttk.Button(
            frame, text="Analyze", command=self.analyze, state="disabled"
        )
        self.analyze_button.grid(
            row=5, column=0, columnspan=2, pady=14, ipadx=24, ipady=5
        )

        results = ttk.LabelFrame(frame, text="Result", padding=12)
        results.grid(row=6, column=0, columnspan=2, sticky="ew")
        results.columnconfigure(1, weight=1)
        ttk.Label(results, text="Predicted class:").grid(row=0, column=0, sticky="w")
        ttk.Label(results, textvariable=self.predicted_class).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        ttk.Label(results, text="Top score:").grid(row=1, column=0, sticky="w")
        ttk.Label(results, textvariable=self.top_score).grid(
            row=1, column=1, sticky="w", padx=(12, 0)
        )
        ttk.Label(results, text="Quality control:").grid(row=2, column=0, sticky="w")
        ttk.Label(results, textvariable=self.quality_result, wraplength=420).grid(
            row=2, column=1, sticky="w", padx=(12, 0)
        )

        ttk.Label(frame, text=SOFTMAX_NOTICE, wraplength=600).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        ttk.Label(frame, textvariable=self.checkpoint_status, wraplength=600).grid(
            row=8, column=0, columnspan=2, sticky="w"
        )

    def _load_checkpoint_once(self) -> None:
        try:
            self.predictor = WastePredictor(CHECKPOINT_PATH)
        except PredictionError as error:
            self.checkpoint_status.set(str(error))
            messagebox.showerror("Checkpoint unavailable", str(error))
            return

        self.checkpoint_status.set(f"Checkpoint loaded: {CHECKPOINT_PATH}")
        self.analyze_button.configure(state="normal")

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a waste image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ),
        )
        if not selected:
            return

        image_path = Path(selected)
        try:
            with Image.open(image_path) as source_image:
                preview = source_image.convert("RGB")
                preview.thumbnail((430, 260), Image.Resampling.LANCZOS)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Image error", f"Could not open image: {error}")
            return

        self.selected_image = image_path
        self.image_path_text.set(str(image_path))
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")
        self.predicted_class.set("—")
        self.top_score.set("—")
        self.quality_result.set("—")

    def analyze(self) -> None:
        if self.predictor is None:
            messagebox.showerror("Checkpoint unavailable", self.checkpoint_status.get())
            return
        if self.selected_image is None:
            messagebox.showinfo("Choose an image", "Select an image before analyzing.")
            return

        try:
            result = self.predictor.predict(
                self.selected_image, self.target_bin.get()
            )
        except PredictionError as error:
            messagebox.showerror("Prediction error", str(error))
            return

        self.predicted_class.set(result.predicted_class)
        self.top_score.set(f"{result.top_score:.1%}")
        self.quality_result.set(result.quality_control_result)


def main() -> None:
    root = tk.Tk()
    WasteClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
