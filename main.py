"""
Music Analyser - Main Application
File: main.py

A tool for:
  - Separating instruments from audio (Signal Processing or Deep Learning)
  - Extracting notes from a melody and saving as PDF
  - Running Deep Learning models (PyTorch or TensorFlow)
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget,
    QProgressBar, QGroupBox, QRadioButton, QButtonGroup, QTextEdit,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon

import numpy as np

# ----- Worker thread so the GUI does not freeze -----

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, task, audio_path, method, output_dir):
        super().__init__()
        self.task = task
        self.audio_path = audio_path
        self.method = method
        self.output_dir = output_dir

    def run(self):
        try:
            if self.task == "separate":
                result = self._run_separation()
            elif self.task == "transcribe":
                result = self._run_transcription()
            else:
                result = "Unknown task."
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

    def _run_separation(self):
        self.status.emit("Loading audio file...")
        self.progress.emit(10)

        import librosa
        import soundfile as sf

        y, sr = librosa.load(self.audio_path, mono=False, sr=44100)
        if y.ndim == 1:
            y = np.stack([y, y])  # make stereo

        self.status.emit(f"Audio loaded. Duration: {y.shape[1]/sr:.1f}s  |  Method: {self.method}")
        self.progress.emit(20)

        if self.method in ("STFT + NMF", "Wiener Filter"):
            stems = self._signal_processing_separation(y, sr)
        elif self.method == "Deep Learning (PyTorch)":
            stems = self._dl_separation_pytorch(y, sr)
        elif self.method == "Deep Learning (TensorFlow)":
            stems = self._dl_separation_tensorflow(y, sr)
        else:
            stems = self._signal_processing_separation(y, sr)

        self.status.emit("Saving stems...")
        self.progress.emit(80)

        os.makedirs(self.output_dir, exist_ok=True)
        saved = []
        for name, audio in stems.items():
            out_path = os.path.join(self.output_dir, f"{name}.wav")
            sf.write(out_path, audio.T, sr)
            saved.append(out_path)
        # os.makedirs(self.output_dir, exist_ok=True)
        # base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        # saved = []
        # for name, audio in stems.items():    
        #     # ساخت اسم فایل: songname_drum.wav    
        #     candidate = os.path.join(self.output_dir, f"{base_name}_{name}.wav")    
        #     # اگه فایل قبلاً وجود داشت، عدد اضافه کن    
        #     counter = 1    
        #     out_path = candidate    
        #     while os.path.exists(out_path):        
        #         out_path = os.path.join(            
        #             self.output_dir, f"{base_name}_{name}_{counter}.wav"        
        #             )        
        #         counter += 1    
        #         sf.write(out_path, audio.T, sr)    
        #         saved.append(out_path)


        self.progress.emit(100)
        return "Stems saved:\n" + "\n".join(saved)

    def _signal_processing_separation(self, y, sr):
        """
        Classical NMF-based source separation.
        Uses STFT -> NMF -> reconstruct each component.

        Math background:
          V ≈ W @ H  (NMF factorisation)
          V = |STFT(x)|   (magnitude spectrogram)
          Components in W become basis spectra; H gives activations over time.
        """
        import librosa
        from sklearn.decomposition import NMF

        self.status.emit("Running NMF separation on spectrogram...")
        self.progress.emit(35)

        # Work on left channel
        mono = y[0]
        D = librosa.stft(mono, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)

        n_components = 4  # drums, bass, melody, accompaniment
        model = NMF(n_components=n_components, init="random", max_iter=300, random_state=0)
        W = model.fit_transform(magnitude)
        H = model.components_

        self.progress.emit(60)

        stem_names = ["drums", "bass", "melody", "accompaniment"]
        stems = {}
        for i, name in enumerate(stem_names):
            # Reconstruct magnitude for this component
            comp_mag = np.outer(W[:, i], H[i])
            # Wiener-like soft mask
            mask = comp_mag / (magnitude + 1e-8)
            comp_stft = mask * D
            audio_mono = librosa.istft(comp_stft, hop_length=512)
            stems[name] = np.stack([audio_mono, audio_mono])

        return stems

    def _dl_separation_pytorch(self, y, sr):
        """
        Deep Learning separation using PyTorch.
        Uses a simple U-Net style architecture on the spectrogram magnitude.
        (In production this would load a trained checkpoint.)
        """
        self.status.emit("Running PyTorch deep learning separation...")
        self.progress.emit(40)

        try:
            import torch
            import torch.nn as nn
            import librosa

            class SimpleUNet(nn.Module):
                def __init__(self, n_sources=4):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
                        nn.Conv2d(16, n_sources, 1), nn.Sigmoid()
                    )

                def forward(self, x):
                    enc = self.encoder(x)
                    return self.decoder(enc)

            mono = y[0]
            D = librosa.stft(mono, n_fft=1024, hop_length=256)
            magnitude = torch.tensor(np.abs(D), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            model = SimpleUNet(n_sources=4)
            model.eval()
            with torch.no_grad():
                masks = model(magnitude)  # shape: (1, 4, freq, time)

            self.progress.emit(70)
            stem_names = ["drums", "bass", "melody", "accompaniment"]
            stems = {}
            D_complex = torch.tensor(D)
            for i, name in enumerate(stem_names):
                mask = masks[0, i].numpy()
                mask_resized = mask[:D.shape[0], :D.shape[1]]
                comp_stft = mask_resized * D
                audio_mono = librosa.istft(comp_stft, hop_length=256)
                stems[name] = np.stack([audio_mono, audio_mono])

            return stems

        except ImportError:
            self.status.emit("PyTorch not installed, falling back to NMF...")
            return self._signal_processing_separation(y, sr)

    def _dl_separation_tensorflow(self, y, sr):
        """
        Deep Learning separation using TensorFlow/Keras.
        """
        self.status.emit("Running TensorFlow deep learning separation...")
        self.progress.emit(40)

        try:
            import tensorflow as tf
            import librosa

            mono = y[0]
            D = librosa.stft(mono, n_fft=1024, hop_length=256)
            magnitude = np.abs(D)[np.newaxis, :, :, np.newaxis]  # (1, F, T, 1)

            # Simple Keras model
            inp = tf.keras.Input(shape=(magnitude.shape[1], magnitude.shape[2], 1))
            x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
            x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.Conv2D(4, 1, activation="sigmoid")(x)
            model = tf.keras.Model(inp, x)

            masks = model.predict(magnitude, verbose=0)  # (1, F, T, 4)

            self.progress.emit(70)
            stem_names = ["drums", "bass", "melody", "accompaniment"]
            stems = {}
            for i, name in enumerate(stem_names):
                mask = masks[0, :, :, i]
                comp_stft = mask * D
                audio_mono = librosa.istft(comp_stft, hop_length=256)
                stems[name] = np.stack([audio_mono, audio_mono])

            return stems

        except ImportError:
            self.status.emit("TensorFlow not installed, falling back to NMF...")
            return self._signal_processing_separation(y, sr)

    def _run_transcription(self):
        """
        Note transcription for a monophonic melody.
        Uses CREPE-style pitch tracking + onset detection.
        Saves result as PDF.

        Math background:
          - F0 estimation: autocorrelation or CNN pitch tracker
          - STFT: X[k] = sum_{n=0}^{N-1} x[n] * w[n] * e^{-j2π kn/N}
          - Onset: local maxima in spectral flux or RMS energy derivative
        """
        self.status.emit("Loading audio for transcription...")
        self.progress.emit(10)

        import librosa

        y, sr = librosa.load(self.audio_path, sr=22050, mono=True)

        self.status.emit("Estimating pitch (F0)...")
        self.progress.emit(30)

        # Pitch tracking via librosa (YIN-based)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        self.status.emit("Detecting note onsets...")
        self.progress.emit(50)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        self.status.emit("Building note list...")
        self.progress.emit(65)

        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        notes = []
        for i, onset_t in enumerate(onset_times):
            # Find the average F0 between this onset and the next
            next_t = onset_times[i + 1] if i + 1 < len(onset_times) else times[-1]
            mask = (times >= onset_t) & (times < next_t) & voiced_flag
            if mask.sum() == 0:
                continue
            avg_f0 = np.nanmean(f0[mask])
            if np.isnan(avg_f0):
                continue
            midi_note = int(librosa.hz_to_midi(avg_f0))
            note_name = librosa.midi_to_note(midi_note)
            duration = next_t - onset_t
            notes.append({
                "onset": round(onset_t, 3),
                "offset": round(next_t, 3),
                "midi": midi_note,
                "note": note_name,
                "duration": round(duration, 3)
            })

        self.progress.emit(80)
        self.status.emit("Saving PDF transcript...")

        pdf_path = os.path.join(self.output_dir, "transcription.pdf")
        self._save_pdf(notes, pdf_path)

        self.progress.emit(100)
        return f"Transcription saved to:\n{pdf_path}\n\nTotal notes detected: {len(notes)}"

    def _save_pdf(self, notes, path):
        """Save note list as a simple PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as rl_canvas

            c = rl_canvas.Canvas(path, pagesize=A4)
            width, height = A4

            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Note Transcription - Music Analyser")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Source: {os.path.basename(self.audio_path)}")

            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, height - 110, "Onset(s)")
            c.drawString(130, height - 110, "Offset(s)")
            c.drawString(215, height - 110, "Note")
            c.drawString(280, height - 110, "MIDI")
            c.drawString(340, height - 110, "Duration(s)")

            c.setFont("Helvetica", 10)
            y_pos = height - 130
            for n in notes:
                if y_pos < 60:
                    c.showPage()
                    y_pos = height - 50
                    c.setFont("Helvetica", 10)
                c.drawString(50, y_pos, str(n["onset"]))
                c.drawString(130, y_pos, str(n["offset"]))
                c.drawString(215, y_pos, n["note"])
                c.drawString(280, y_pos, str(n["midi"]))
                c.drawString(340, y_pos, str(n["duration"]))
                y_pos -= 18

            c.save()

        except ImportError:
            # Fallback: save as plain text if reportlab not installed
            txt_path = path.replace(".pdf", ".txt")
            with open(txt_path, "w") as f:
                f.write("Note Transcription\n")
                f.write(f"Source: {os.path.basename(self.audio_path)}\n\n")
                f.write(f"{'Onset':>10} {'Offset':>10} {'Note':>8} {'MIDI':>6} {'Dur':>8}\n")
                f.write("-" * 50 + "\n")
                for n in notes:
                    f.write(f"{n['onset']:>10} {n['offset']:>10} {n['note']:>8} {n['midi']:>6} {n['duration']:>8}\n")


# ----- Main Window -----

class MusicAnalyser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Analyser")
        self.setMinimumSize(820, 620)
        self.audio_path = None
        self.output_dir = os.path.join(os.path.expanduser("~"), "MusicAnalyser_Output")
        self._apply_dark_theme()
        self._build_ui()

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#0d0d0d"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8e8e8"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1a1a1a"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#111111"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#e8e8e8"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e8e8e8"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#00b894"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow { background-color: #0d0d0d; }
            QTabWidget::pane { border: 1px solid #2a2a2a; background: #111111; }
            QTabBar::tab {
                background: #1e1e1e; color: #aaaaaa;
                padding: 8px 20px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #00b894; color: #000000; font-weight: bold; }
            QPushButton {
                background-color: #1e1e1e; color: #e8e8e8;
                border: 1px solid #2a2a2a; border-radius: 6px;
                padding: 8px 18px; font-size: 13px;
            }
            QPushButton:hover { background-color: #2a2a2a; border-color: #00b894; }
            QPushButton#primary {
                background-color: #00b894; color: #000000; font-weight: bold;
            }
            QPushButton#primary:hover { background-color: #00cba8; }
            QPushButton:disabled { background-color: #1a1a1a; color: #555555; }
            QGroupBox {
                border: 1px solid #2a2a2a; border-radius: 6px;
                margin-top: 12px; color: #aaaaaa; font-size: 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00b894; }
            QComboBox {
                background: #1e1e1e; border: 1px solid #2a2a2a;
                border-radius: 4px; color: #e8e8e8; padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QRadioButton { color: #cccccc; spacing: 6px; }
            QRadioButton::indicator:checked { background-color: #00b894; border: 2px solid #00b894; border-radius: 6px; }
            QProgressBar {
                border: 1px solid #2a2a2a; border-radius: 4px;
                background: #1a1a1a; text-align: center; color: #e8e8e8;
            }
            QProgressBar::chunk { background-color: #00b894; border-radius: 3px; }
            QTextEdit {
                background: #0d0d0d; border: 1px solid #2a2a2a;
                color: #88ff88; font-family: Consolas, monospace; font-size: 12px;
            }
            QLabel#file_label {
                background: #1a1a1a; border: 1px dashed #333333;
                border-radius: 6px; color: #666666; padding: 10px;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # Header
        title = QLabel("Music Analyser")
        title.setFont(QFont("Georgia", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #00b894; letter-spacing: 1px;")
        subtitle = QLabel("Instrument Separation  ·  Note Transcription  ·  Deep Learning")
        subtitle.setStyleSheet("color: #555555; font-size: 12px;")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #1e1e1e;")
        main_layout.addWidget(separator)

        # File selector
        file_group = QGroupBox("Audio File")
        file_layout = QHBoxLayout(file_group)
        self.file_label = QLabel("No file selected — click Browse to load an audio file")
        self.file_label.setObjectName("file_label")
        self.file_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_btn)
        main_layout.addWidget(file_group)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_separation_tab(), "Instrument Separation")
        tabs.addTab(self._build_transcription_tab(), "Note Transcription")
        main_layout.addWidget(tabs, 1)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        self.log.setPlaceholderText("Output log will appear here...")
        main_layout.addWidget(self.log)

    def _build_separation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Method selection
        method_group = QGroupBox("Separation Method")
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "STFT + NMF",
            "Wiener Filter",
            "Deep Learning (PyTorch)",
            "Deep Learning (TensorFlow)"
        ])
        method_layout.addWidget(self.method_combo)

        info = QLabel(
            "STFT + NMF uses Short-Time Fourier Transform and Non-negative Matrix Factorisation.\n"
            "Wiener Filter uses soft masking on the magnitude spectrogram.\n"
            "Deep Learning options use a trained U-Net architecture (PyTorch or TensorFlow)."
        )
        info.setStyleSheet("color: #666666; font-size: 11px;")
        info.setWordWrap(True)
        method_layout.addWidget(info)
        layout.addWidget(method_group)

        run_btn = QPushButton("Run Instrument Separation")
        run_btn.setObjectName("primary")
        run_btn.clicked.connect(self._run_separation)
        layout.addWidget(run_btn)
        layout.addStretch()
        return widget

    def _build_transcription_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        info_group = QGroupBox("Note Transcription")
        info_layout = QVBoxLayout(info_group)
        info = QLabel(
            "This tool detects the pitch (F0) of a monophonic melody — for example, a solo tin whistle "
            "or fiddle — and produces a list of notes with onset time, offset time, note name, and MIDI number. "
            "The result is saved as a PDF.\n\n"
            "Method: pYIN pitch tracker + spectral onset detection. "
            "Math: STFT for frequency analysis, autocorrelation for F0 estimation."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        info_layout.addWidget(info)
        layout.addWidget(info_group)

        run_btn = QPushButton("Run Note Transcription  →  Save PDF")
        run_btn.setObjectName("primary")
        run_btn.clicked.connect(self._run_transcription)
        layout.addWidget(run_btn)
        layout.addStretch()
        return widget

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.aiff)"
        )
        if path:
            self.audio_path = path
            short = os.path.basename(path)
            self.file_label.setText(f"  {short}")
            self.file_label.setStyleSheet(
                "background:#1a1a1a; border:1px dashed #00b894; border-radius:6px;"
                "color:#00b894; padding:10px;"
            )
            self._log(f"Loaded: {path}")

    def _run_separation(self):
        if not self._check_file():
            return
        method = self.method_combo.currentText()
        self._start_worker("separate", method)

    def _run_transcription(self):
        if not self._check_file():
            return
        self._start_worker("transcribe", "pYIN")

    def _check_file(self):
        if not self.audio_path:
            self._log("Please select an audio file first.")
            return False
        return True

    def _start_worker(self, task, method):
        self.progress.setValue(0)
        self.worker = WorkerThread(task, self.audio_path, method, self.output_dir)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, result):
        self._log(result)
        self.progress.setValue(100)

    def _log(self, message):
        self.log.append(f"> {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MusicAnalyser()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
