# Kutub Automator — Multimodal Arabic ASR & Document OCR Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA NeMo](https://img.shields.io/badge/ASR-NVIDIA%20NeMo%20FastConformer-76B900.svg)](https://github.com/NVIDIA/NeMo)
[![OCR](https://img.shields.io/badge/OCR-Surya%20%7C%20Qari-orange.svg)](https://github.com/VikParuchuri/surya)
[![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen.svg)](tests/)

An end-to-end multi-source data ingestion, speech-to-text (ASR), document OCR parsing, and text synthesis platform built with Python, PyTorch, NVIDIA NeMo, Hugging Face VLMs, Surya OCR, and Gradio.

`Kutub Automator` automates the collection, document parsing, transcription, cleaning, and indexing of complex media streams and historical book scans.

---

## 🚀 Complete Feature Breakdown

### 1. 🎙️ High-Precision Arabic Speech-to-Text (ASR) Engine

* **NVIDIA NeMo FastConformer Integration (`modules/transcription_engine.py`):**
  * Local GPU inference using NVIDIA NeMo's `stt_ar_fastconformer_hybrid_large_pc_v1.0` model (424M parameters).
  * Audio pre-processing pipeline powered by `pydub`, `soundfile`, and `librosa`.
  * Automatic audio splitting and chunking to handle multi-hour lectures without CUDA memory overflow.
  * Export capabilities for raw text, time-coded SRT subtitles, and structured JSON logs.
  * Automated model downloader via `python download_model.py`.

---

### 2. 📄 Document Parsing & Vision-Language OCR

* **Surya Document OCR & Layout Server (`start_surya_server.ps1`):**
  * Local high-throughput OCR server integration running Surya for text line detection, reading order sorting, and layout recognition.
  * Powers comparative evaluation between models via `modules/best_ocr.py` and `modules/ocr_compare.py`.

* **Qari Arabic OCR Engine (`modules/qari_ocr.py`):**
  * Specialized OCR fallback engine tailored for printed and historical Arabic text layouts.

* **PyMuPDF PDF Renderer (`modules/unlimited_ocr.py`):**
  * Converts PDF document pages into high-DPI image caches on the fly using `fitz` (PyMuPDF).
  * Batch processing with safety caps (max 15 pages per batch) to prevent model hallucination.

* **Baidu Unlimited-OCR VLM Integration:**
  * Runs Baidu's `Unlimited-OCR` Vision-Language Model locally in `bfloat16` precision.
  * **Live Token Streaming Patch:** Custom monkey-patching of `TPSTextStreamer` for real-time live token streaming to the Gradio web UI.
  * **Regex Cleaning Pipeline:** Automatically strips VLM coordinate tags (`<|det|>...<|/det|>`), reference markers, and cleans page breaks.

---

### 3. 📥 Multi-Source Data Ingestion & Scraping

* **Telegram Channel Crawler (`modules/telegram_scraper.py`):**
  * Asynchronous crawler built on `Telethon` for crawling target Telegram channels and groups.
  * Automated file discovery (Audio, Video, PDF scans) with workspace caching to eliminate duplicate downloads.
  * Interactive authentication workflow with persistent session management.

* **Facebook Video Downloader (`modules/fb_downloader.js`):**
  * Node.js helper script for extracting video media streams from Facebook links.

---

### 4. 🔊 Text-To-Speech (TTS) Synthesis

* **Higgs & Expressive TTS Engine (`modules/higgs_tts.py`):**
  * Integrated speech synthesis module for generating fluid audio output from transcribed text.

---

### 5. 🖥️ Interactive Control Center & System Engineering

* **Gradio Web Dashboard (`app.py`):**
  * Unified web dashboard with tabbed interfaces for Telegram Scraping, Audio ASR, Unlimited-OCR, Surya OCR, Qari OCR, and TTS.

* **Windows & System Stability Patches:**
  * Low-level SSL context patching to bypass certificate errors on local networks.
  * Process-wide temporary directory file lock cleanup patches to prevent `WinError 32` file locking on Windows.
  * High-precision logging with file and line tracking.

---

## 🛠️ System Architecture

```mermaid
graph TD
    subgraph Ingestion Layer
        A1[Telegram Channels] -->|Telethon Scraper| B[Workspace Download Cache]
        A2[Facebook Videos] -->|Node.js Downloader| B
        A3[Local PDF Scans & Audio] --> B
    end

    subgraph Processing Engine
        B -->|Audio Streams| C[Audio Splitter & Resampler]
        C --> D[NVIDIA NeMo FastConformer ASR]
        
        B -->|PDF Documents| E[PyMuPDF Renderer]
        E -->|Page Images| F[Surya OCR Server]
        E -->|Page Images| G[Qari OCR Engine]
        E -->|Page Images| H[Baidu Unlimited-OCR VLM]
        
        F --> I[Multi-Engine Comparator & Selector]
        G --> I
        H --> I
    end

    subgraph Post-Processing & Output
        D --> J[SRT Subtitles / Timed Logs]
        I --> K[Clean Normalized Text]
        K -->|Optional| L[TTS Speech Synthesis]
    end

    subgraph User Interface
        M[Gradio Multi-Tab Web UI] <--> Ingestion Layer
        M <--> Processing Engine
        M <--> Post-Processing & Output
    end
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive test suite covering OCR output isolation, temporary lock resilience, and multi-engine comparison:

```bash
# Run the 17 automated tests
pytest
```

**Test results:** `17 passed in 4.26s (100% success)`

---

## 💻 Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch (CUDA), Hugging Face Transformers.
* **Specialized ML Toolkits:** NVIDIA NeMo Toolkit (`nemo_toolkit[asr]`), Surya OCR, Qari OCR, Baidu Unlimited-OCR.
* **Document & Image Processing:** PyMuPDF (`fitz`), Pillow, OpenCV, Librosa, SoundFile, Pydub.
* **Web UI & Networking:** Gradio, Telethon (Telegram API), Requests, HTTPX, Node.js.

---

## ⚙️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/h-yassin-ai/Kutub-Automator.git
   cd Kutub-Automator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the NVIDIA NeMo FastConformer model (424 MB):**
   ```bash
   python download_model.py
   ```

4. **Launch the dashboard:**
   ```bash
   python app.py
   ```
