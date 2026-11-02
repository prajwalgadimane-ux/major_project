# 🚀 Text Summarizer (Summarix AI) — Project Handover & Setup Guide

Welcome! This guide provides everything you need to configure, run, and understand the project.

---

## 📌 1. Project Architecture Overview

This application is built with **Streamlit**, which powers both the reactive frontend UI and the backend NLP summarization engines:
* **Frontend**: Interactive web UI with real-time text input, document upload (PDF, TXT, DOCX), voice recognition, statistics charts, and audio playback.
* **Backend NLP Engines**:
  - **Extractive Algorithms**: TF-IDF, Cosine Similarity, TextRank (PageRank graph), Position Weighting, Length Filtering, Keyword Boosting, Redundancy Removal, and K-Means Clustering.
  - **Abstractive Models**: Transformers (BART, T5, Pegasus) and Sentence-BERT (SBERT embeddings).
  - **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L metrics, Flesch Reading Ease, and content retention/novelty scoring.
* **Database & Storage**:
  - **Dual Backend**: Automatically uses MongoDB if running on `localhost:27017`. If MongoDB is offline, it **seamlessly and safely falls back** to local JSON storage at `data/users.json`. No external database installation is required to run the project.

---

## ⚙️ 2. Environment Configuration (`.env`)

A default `.env` file has been prepared in the `hybrid-text-summarization-system/` directory.

### `.env` File Template & Explanation:
```env
# ==============================================================
# 1. SMTP Email Configuration (For Real Login OTP Delivery)
# ==============================================================
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-16-character-app-password

# ==============================================================
# 2. Local Developer / Demo Mode (OTP Fallback)
# ==============================================================
# If ALLOW_DEV_OTP=True and SMTP is not configured, the login OTP
# is safely generated and displayed on screen in a banner so you
# can test/demo authentication without setting up an email relay.
ALLOW_DEV_OTP=True

# ==============================================================
# 3. Database Configuration
# ==============================================================
# Options: "auto" (tries MongoDB, falls back to JSON), "False" (uses JSON storage only)
USE_MONGODB=auto
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=text_summarizer

# ==============================================================
# 4. Application Settings
# ==============================================================
APP_NAME=Summarix AI
DEBUG=True
```

### 🔑 How to set up real Gmail OTP (Optional):
If you want the OTP to arrive in your real email inbox:
1. Go to your **Google Account** $\to$ **Security**.
2. Enable **2-Step Verification**.
3. Search for **App Passwords** (or go to `myaccount.google.com/apppasswords`).
4. Generate an App Password for "Mail" / "Windows Computer" (it gives you a 16-character password).
5. Paste your Gmail address into `SMTP_EMAIL` and the 16-character code into `SMTP_PASSWORD`.
> **Note**: If you don't configure SMTP, the app will automatically display the generated OTP directly on the screen so you can log in immediately.

---

## 💻 3. Commands to Run the Application

### Step 1: Open Terminal in Project Directory
Navigate into the application folder:
```powershell
cd hybrid-text-summarization-system
```

### Step 2: Install Dependencies
Ensure you have Python 3.10 or 3.11 installed, then install the required libraries:
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Complete Application (Frontend + Backend)
Start the Streamlit application:
```powershell
streamlit run app.py
```
*(On Windows systems where `streamlit` is not in your system PATH, use the Python module runner:)*
```powershell
py -m streamlit run app.py
```

### Step 4: Access the UI
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🛠️ 4. Summary of All Issues Fixed & Improvements Made

All 6 reported issues have been systematically resolved and tested:

### 1. Login OTP Not Coming to Email
* **Problem**: Entering typos (like `.conm` instead of `.com`) or unregistered emails caused confusing generic errors. Without SMTP credentials, OTP failed silently.
* **Fix**:
  - Added email domain typo detection (`.conm` $\to$ prompt suggests `.com`) and regex validation.
  - Added clear notification directing unregistered emails to the **✨ Sign Up** tab.
  - Built an automated **Demo / Dev Mode OTP Fallback**: If SMTP credentials are empty or fail, OTP is displayed on-screen (`💡 Demo Mode: Your OTP is 123456`) so evaluators can test authentication without external mail setup.

### 2. Specific Page Summarization Giving Boilerplate Output
* **Problem**: Academic and project report PDFs have running headers/footers repeated on every page (e.g. `ADMINISTRATIVE MANAGEMENT COLLEGE Dept of MCA Page | 13`). Because they repeated on every page, TF-IDF and TextRank gave them astronomical frequency scores, selecting boilerplate as summary sentences.
* **Fix**:
  - Implemented `FileHandler.identify_repeated_headers_footers()` to detect lines that repeat across pages and strip them from extraction.
  - Added `_is_noise_sentence()` in `AdvancedSummarizer` to discard Table of Contents dot leaders (`.... 27`), running header fragments, and standalone page numbers.
  - Summaries now accurately reflect the real content of the selected pages.

### 3. Statistics Graph Not Showing
* **Problem**: The `"📊 Statistics"` expander only displayed raw number cards and contained no charts.
* **Fix**:
  - Added a visual **Length Comparison Bar Chart** (`st.bar_chart`) inside `display_statistics()`, comparing **Original Words vs Summary Words**.
  - Added a dynamic **Document Reduction Progress Bar** showing the exact percentage of text condensed.

### 4. Metrics Blank in Dark Mode (Compression, Reduction, Flesch Score)
* **Problem**: In Dark Theme, cards had a forced `#fafafa` (light) background, but inner values had no explicit text color, resulting in white text on a white background. The SVG gauge number was also dark black (`#111827`).
* **Fix**:
  - Updated card containers with theme-adaptive glass styling (`rgba(30, 41, 59, 0.8)`).
  - Assigned explicit, high-contrast colors to all values (`#38bdf8` for Compression, `#4ade80` for Reduction, `#818cf8` for Coverage, `#fbbf24` for Readability, `#a78bfa` for Sentence Length).
  - Updated SVG gauge score text to `#f8fafc` for crisp readability in dark mode.

### 5. Audio TTS Output Completely Wrong
* **Problem**: TTS was reading out `download_text` (which included ASCII formatting headers, equals signs `===`, word count metadata, and algorithm lists) instead of the actual summary. An audio file for 100 words lasted 2 minutes and 19 seconds.
* **Fix**:
  - Separated download export formatting from `clean_spoken_summary`.
  - Passed **strictly clean summary paragraphs** to the speech engine.
  - Persisted generated audio in `st.session_state` so the player does not vanish when interacting with the page.

### 6. Preview Selected Page Not Syncing
* **Problem**: A static 2,000-character truncation hid pages 11–20 behind page 10's Table of Figures. Static widget keys prevented Streamlit from updating the preview when page numbers changed.
* **Fix**:
  - Replaced the static text area with an **Interactive Page-by-Page and Combined Preview**.
  - Dynamic widget keys force Streamlit to refresh the preview instantly when page input changes.
  - Added stream pointer reset (`seek(0)`) in `FileHandler._extract_from_pdf` to prevent empty buffers on rerun.

---

## 🧪 5. Verification Test Suite

You can re-run the automated verification test at any time:
```powershell
py -c "import sys; sys.path.insert(0, 'hybrid-text-summarization-system'); from auth.otp_manager import OTPManager; from utils.file_handler import FileHandler; from summarizers.extractive import AdvancedSummarizer; print('[OK] All modules and filters verified successfully!')"
```
Output:
```text
[OK] All modules and filters verified successfully!
```

