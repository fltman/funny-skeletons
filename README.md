# Funny Skeletons

[![Support me on Patreon](https://img.shields.io/badge/Patreon-Support%20my%20work-FF424D?style=flat&logo=patreon&logoColor=white)](https://www.patreon.com/AndersBjarby)

A Halloween webcam art installation. Two skeletons, Knotan and Skallan, watch your garden through the camera and crack jokes about whatever they see, Statler-and-Waldorf style. It takes a photo at intervals, only reacts when it detects people, has GPT-4o describe and comment on the scene in Swedish, and speaks the lines aloud with separate ElevenLabs voices for each skeleton.

## What it does

- Lists available cameras (via `imagesnap`) and lets you pick one
- Snaps a photo every few seconds and runs OpenCV face detection
- Skips and deletes photos with no people in them
- Sends photos with people to GPT-4 Vision for a description, then asks GPT-4o to improvise a short, funny two-skeleton dialogue (in Swedish)
- Generates speech per line with ElevenLabs (a distinct voice for Knotan and Skallan) and plays it back through pygame

## Requirements

- macOS with [`imagesnap`](https://github.com/rharder/imagesnap) installed (`brew install imagesnap`)
- A webcam
- Python packages: `openai`, `requests`, `python-dotenv`, `pygame`, `opencv-python`
- A `.env` file with `OPENAI_API_KEY` and `ELEVENLABS_KEY`

## Setup

```bash
pip install openai requests python-dotenv pygame opencv-python
brew install imagesnap

# .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ELEVENLABS_KEY=..." >> .env
```

## Usage

```bash
python macos_webcam_interval_photos.py
```

Select a camera when prompted; the script then runs continuously, photographing, commenting, and speaking until you stop it.

## Tech

Python, OpenAI (GPT-4 Vision + GPT-4o), ElevenLabs text-to-speech, OpenCV face detection, pygame audio, imagesnap.
