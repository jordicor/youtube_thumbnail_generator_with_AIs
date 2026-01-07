# YouTube Thumbnail Generator

### AI-Powered Thumbnails, Titles & Descriptions with Face Identity Preservation

> **From 100 videos uploaded in 3 years... to batch-processing 178 videos in days.**
>
> *Built in one week with Claude Code during my "vacation" from entrepreneurship.*

---

## The Story Behind This Tool

Back in 2015-2016, I was the manager of **FocusingsVlogs** (Mel Dominguez), a Spanish YouTuber who had built an audience of 240,000+ subscribers. Life happened, and she deleted her channel for personal reasons before the pandemic.

I had a backup of almost all her videos—but not the thumbnails.

Fast forward to 2023. I reached out to Mel with an idea: let's re-upload her old videos as a digital memory archive. She agreed, as long as I handled everything so she could stay off the radar.

**The problem?** Creating thumbnails for 278 videos when:
- The original thumbnails were gone
- The videos were 720p with camera noise and low sharpness
- Mel had changed her appearance significantly since 2012-2016
- Each thumbnail required: watching the video, finding good frames, upscaling, retouching, designing in Canva
- AI image generation existed but couldn't reliably maintain her face

After 3 years of manual work (2023-2025), I had only uploaded 100 videos. **178 still waiting.**

Then came the AI revolution of late 2025. Models could finally:
- Generate high-quality images with text
- Preserve facial identity from reference photos
- Understand complex creative prompts

**So I built this tool in 8 days during the holidays** (December 27, 2025 - January 4, 2026) using Claude Code. Not even consecutive days—New Year's happened in the middle. And yet here we are: complete web interface, face detection pipeline, multi-provider AI integration. I'm genuinely amazed at how fast we can ship full applications now.

It automates what used to take me hours per video into a streamlined, AI-powered pipeline.

---

## What This Tool Does

```
VIDEO FILE → Scene Detection → Face Extraction → Face Clustering → Transcription
                                                                        ↓
THUMBNAILS ← Image Generation ← Creative Prompts ← LLM Analysis ← Content Understanding
```

**One click** transforms any video into multiple AI-generated thumbnail options, each preserving the exact facial identity of the person you select.

---

## Key Features

### Identity Cloning Technology
The system uses **strict identity preservation prompts** that prevent AI models from "averaging" or "beautifying" faces. Your reference person appears in thumbnails looking exactly like themselves—not a similar-looking AI interpretation.

```
"A friend of this person should INSTANTLY recognize them in the generated image."
```

### Multi-Provider AI Support
Switch between image generation providers based on your needs:

| Provider | Best Models | Max References | Strengths |
|----------|-------------|----------------|-----------|
| **Google Gemini** | gemini-3-pro-image-preview | 14 images | Best identity preservation |
| **OpenAI** | gpt-image-1.5 | 16 images | High fidelity |
| **Poe** | nanobananapro, flux2pro | 14 images | Fast, reliable |
| **Replicate** | FLUX 1.1 Pro | 1 image | Open-source option |

### Automatic Face Clustering
No manual face labeling needed. The system:
1. Detects all faces in the video using InsightFace
2. Creates 512-dimensional embeddings for each face
3. Clusters faces by identity using DBSCAN
4. Separates clusters by scene (handles costumes/disguises)

### Smart Frame Selection
Not all frames are equal. The system scores each face by:
- **Quality**: Sharpness, brightness, contrast
- **Pose**: Front-facing preferred over profiles
- **Expression**: Balanced selection (smiling, neutral, mouth closed)
- **Size**: Larger faces score higher

### Creative Prompt Generation
Uses LLMs (Claude, GPT-4, Gemini) to analyze video transcripts and generate:
- Multiple creative concepts per video
- Suggested clickable titles
- Color palettes and mood descriptions
- Text overlay suggestions

### AI Title & Description Generation
Beyond thumbnails, the tool generates YouTube-ready metadata:
- **Titles**: Multiple options in different styles (neutral, SEO-optimized, clickbait)
- **Descriptions**: With optional timestamps, hashtags, and calls-to-action
- **Title-aware thumbnails**: When you select a title, the image generation takes it into account to create visually coherent thumbnails that complement your chosen title

### Real-Time Progress Tracking
Server-Sent Events (SSE) provide instant feedback:
- Watch scene detection progress
- See faces being extracted live
- Track thumbnail generation in real-time

---

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **FFmpeg** installed and in PATH
- **NVIDIA GPU** recommended (works on CPU, 5-10x slower)
- API keys for at least one provider (Gemini, OpenAI, or Poe)

### Installation

```bash
# Clone the repository
git clone https://github.com/jordicor/youtube_thumbnail_generator.git
cd youtube_thumbnail_generator

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

### Configuration

Edit `.env` with your settings:

```bash
# Required: At least one image provider
GEMINI_API_KEY=your-gemini-key
# OPENAI_API_KEY=your-openai-key
# POE_API_KEY=your-poe-key

# Required: At least one LLM for prompts
ANTHROPIC_API_KEY=your-anthropic-key
# OPENAI_API_KEY=your-openai-key

# Directories
VIDEOS_DIR=C:/path/to/your/videos
OUTPUT_DIR=./output

# Image generation settings
IMAGE_PROVIDER=gemini
GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview

# Transcription (local Whisper is free)
USE_LOCAL_WHISPER=true
WHISPER_MODEL=turbo
TRANSCRIPTION_LANGUAGE=en
```

### Gran Sabio LLM Setup (Required)

The web interface uses [**Gran Sabio LLM**](https://github.com/jordicor/GranSabio_LLM) for intelligent prompt generation. This is a multi-provider AI orchestration engine that handles all LLM calls with unified API key management.

**1. Clone Gran Sabio LLM:**
```bash
git clone https://github.com/jordicor/GranSabio_LLM.git
cd GranSabio_LLM
pip install -r requirements.txt
```

**2. Configure Gran Sabio (add your API keys):**
```bash
cp .env.template .env
# Edit .env with your API keys (OpenAI, Anthropic, Google, etc.)
```

**3. Start Gran Sabio server:**
```bash
python main.py
# Server starts at http://localhost:8000
```

**4. Configure this project to use Gran Sabio:**

Add to your `.env`:
```bash
GRANSABIO_CLIENT_PATH=C:/path/to/GranSabio_LLM/client
GRANSABIO_LLM_URL=http://localhost:8000
```

> **Why Gran Sabio?** Instead of duplicating API integration code, Gran Sabio provides a unified interface to multiple AI providers (OpenAI, Anthropic, Google, xAI, OpenRouter) with features like multi-model QA, thinking modes, and automatic retries. Your API keys are configured once in Gran Sabio and shared across projects.

### Run the Web UI

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

---

## Usage Guide

### Step 1: Add Your Videos

1. Click the directory dropdown
2. Add a new directory pointing to your video folder
3. Click "Scan" to detect videos

### Step 2: Analyze Videos

1. Select videos (or use "Select All")
2. Click "Analyze"
3. Wait for the pipeline to complete:
   - Scene detection (~30s per 10min video)
   - Face extraction (~2min per 1000 frames)
   - Face clustering (~5s)
   - Transcription (~1min per 10min audio)

### Step 3: Configure Generation

1. Click on an analyzed video
2. Generate titles and descriptions using the AI tabs (optional but recommended)
3. Review detected face clusters
4. **Star** the cluster representing the person for thumbnails
5. Select your preferred title—the AI will design thumbnails that complement it
6. Optionally:
   - Add custom instructions
   - Upload a style reference image
   - Adjust concepts and variations count

### Step 4: Generate Thumbnails

1. Click "Generate Thumbnails"
2. Watch as AI creates your options
3. Download favorites or the full ZIP

---

## Advanced Features

### Person + Scene Clustering

Videos with costumes, disguises, or location changes can confuse simple face clustering. The **"View by Person + Scene"** mode subdivides each person into scene-specific clusters:

- Person 1 - Scene 1 (regular outfit)
- Person 1 - Scene 5 (pirate costume)
- Person 1 - Scene 12 (different location)

### Manual Reference Selection

Auto-selected frames not ideal? The Reference Frame Manager lets you:
- Browse all extracted frames
- Filter by scene
- Drag to reorder priority
- Add/remove from AI references

### External Style Reference

Upload an image for style inspiration:
- The AI analyzes composition, colors, lighting
- Your style reference influences the output
- Identity preservation remains priority #1

### CLI Batch Processing

For power users processing many videos:

```bash
# Process all videos in directory
python main.py

# Single video
python main.py --single "my_video.mp4"

# Custom generation settings
python main.py --num-prompts 5 --num-variations 3 --image-provider poe

# Force regeneration
python main.py --force-thumbnails
```

---

## Architecture

```
youtube_thumbnail_generator/
├── api/                    # FastAPI web server
│   ├── main.py             # App initialization
│   └── routes/             # REST endpoints
├── services/               # Business logic
│   ├── analysis_service.py # Analysis pipeline (1600+ lines)
│   └── generation_service.py # Generation pipeline
├── templates/              # Jinja2 HTML (dark theme UI)
├── static/                 # CSS + JavaScript
├── database/               # SQLite with async access
└── output/                 # Generated files per video
```

For detailed technical documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Requirements

### Hardware
- **Minimum**: 8GB RAM, any modern CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM

### Software
- Python 3.10-3.11
- FFmpeg (for audio extraction)
- Windows 10/11 (primary target), Linux/Mac should work

### API Costs (Approximate)
- **Gemini**: Free tier available, ~$0.01-0.05 per thumbnail
- **OpenAI GPT Image**: ~$0.02-0.08 per thumbnail
- **Poe**: Subscription-based, varies by model
- **Whisper Local**: Free (uses your GPU/CPU)

---

## Who Is This For?

- **YouTubers re-uploading old content** without original thumbnails
- **Channel managers** handling multiple videos
- **Content creators** who forgot to shoot thumbnail photos
- **Anyone** who wants AI-generated thumbnails that actually look like them

---

### Other Projects
- [**Acerting Art**](https://www.youtube.com/@AcertingArt): 430K+ subscribers, relaxation/meditation music
- [**GranSabio LLM**](https://github.com/jordicor/GranSabio_LLM): Multi-layer QA system for LLM content generation
- [**VR Relaxation Space Room**](https://jordicor.itch.io/acerting-art-vr-relaxation-space-room): VR meditation app (Unity/Maya)
- [**Neo Atlantis**](https://github.com/jordicor/neo-atlantis): RPG game with 200+ 3D models (1999)
- [**Security Research Archive**](https://github.com/jordicor/security-research-archive): 6 vulnerabilities, 2 CVEs (2004-2006)

---

## License

MIT License - Use freely, attribution appreciated.

---

## Acknowledgments

- **Mel (FocusingsVlogs)** for trusting me with her content archive
- **Claude Code** for making a week-long dev sprint possible
- **ElevenLabs** for their excellent speech-to-text API with speaker diarization
- The open-source community behind InsightFace, PySceneDetect, and Whisper

---

<p align="center">
  <i>Built with obsession, automation, and a healthy disregard for manual labor.</i>
</p>

<p align="center">
  <a href="https://github.com/jordicor">GitHub</a> •
  <a href="https://jordicor.com">Website</a> •
  <a href="https://www.youtube.com/@AcertingArt">YouTube</a>
</p>
