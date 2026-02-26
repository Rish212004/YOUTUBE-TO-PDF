Purpose: Converts YouTube video audio into transcribed text documents (PDFs)

Key Features:

Batch Processing: Reads YouTube URLs from a CSV file
Audio Download: Uses yt_dlp to download best quality audio from YouTube videos
AI Transcription: Uses Whisper (via HuggingFace Transformers) to convert speech-to-text
PDF Export: Saves transcribed text as PDF files
GPU Optimization: Auto-detects available hardware (MPS for Mac, CUDA for NVIDIA, CPU fallback)
Smart CSV Parsing: Intelligently detects URL and filename columns regardless of order
Workflow:

User selects a CSV file with YouTube URLs
Script downloads audio from each video
Whisper AI model transcribes the audio
Saves transcription as a PDF file
Cleans up temporary files
