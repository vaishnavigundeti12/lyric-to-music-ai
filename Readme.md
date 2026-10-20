Here is a professional, comprehensive `README.md` for your GitHub repository. It highlights the technical depth of your project, which is perfect for your CV and for your professor to review.

Create a file named `README.md` in your project folder, paste this inside, save it, and then push it to GitHub.

***

# 🎵 Intelligent Lyric-to-Music Style Recommendation & Generation

An end-to-end AI pipeline that bridges the gap between raw lyrics and AI music generation. Instead of requiring users to know complex musical terminology, this system uses **Large Language Models (LLMs)** to analyze lyrics, combine them with a user's text-based style request, and automatically generate an optimized prompt for instrumental audio generation.

## 🚀 Key Features

*   **User-Driven Style Constraint:** Users provide both lyrics and a plain-text style request (e.g., "slow indie folk with acoustic guitar").
*   **5-Step Chain-of-Thought (CoT):** The LLM explicitly reasons through the emotional tone, themes, and the reconciliation of the user's style request before outputting a structured JSON blueprint.
*   **Visible Reasoning:** The LLM's exact thought process is formatted and printed to the terminal for transparency and debugging.
*   **Local Music Generation:** Utilizes Meta's `musicgen-small` to generate a 5-second instrumental locally on CPU.
*   **Semantic Song Recommendations:** Uses `Sentence-BERT` embeddings and cosine similarity to recommend the top 3 famous songs that match the generated vibe.
*   **Security & Evaluation:** Includes a Regex-based prompt injection filter and an optional "LLM-as-Judge" module to score the mood accuracy and genre fit.

## 🛠️ Tech Stack

*   **Language:** Python 3.13
*   **LLM API:** OpenRouter (via OpenAI SDK)
*   **Audio Generation:** Meta MusicGen (`transformers`, `torch`)
*   **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Web Interface:** Gradio
*   **Audio Processing:** SoundFile, NumPy

## 📂 Repository Structure

*   `app.py`: The main Gradio web application and core pipeline logic.
*   `build_song_db.py`: Generates the style profiles for the recommendation database.
*   `build_song_audio.py`: Pre-generates the 5-second audio previews for the recommended songs.
*   `requirements.txt`: Pinned dependencies for reproducible environments.

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
You need an API key from [OpenRouter.ai](https://openrouter.ai) to access the free LLM.
Create a `.env` file or set it directly in your terminal:
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-v1-your_api_key_here"

# Mac/Linux
export OPENROUTER_API_KEY="sk-or-v1-your_api_key_here"
```

### 5. Build the Database (First run only)
Before running the app for the first time, generate the song style database and the preview audio files:
```bash
python build_song_db.py
python build_song_audio.py
```
*(Note: `build_song_audio.py` will take 5-10 minutes to run on CPU as it generates the preview WAV files).*

## 🎮 Usage

Start the application:
```bash
python app.py
```

1. Open your browser and go to `http://127.0.0.1:7860`.
2. Paste your lyrics into the left text box.
3. Enter your desired music style (e.g., "upbeat electronic pop").
4. Click **Analyze & Generate**.
5. **Watch your VS Code terminal** to see the 5-step Chain-of-Thought reasoning, followed by the generated audio player and song recommendations in the UI.

## 🧠 How the Prompt Engineering Works

The system uses a strict JSON constraint prompt combined with Chain-of-Thought:
1.  **Deconstruct:** What did the user ask for?
2.  **Analyze:** What do the lyrics actually mean?
3.  **Reconcile:** If the user wants a "happy pop song" but the lyrics are about "heartbreak," how do we blend them? (e.g., "Fast tempo but minor key").
4.  **Commit:** Output the final 7-field JSON schema.

## 📄 License
This project is created for academic purposes (Foundation of AI & Prompt Engineering Microproject). Models used are subject to their respective open-source licenses (Meta Llama, MusicGen).