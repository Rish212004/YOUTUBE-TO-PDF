import os
import csv
import warnings
import tkinter as tk
from tkinter import filedialog

# --- 🛑 CRITICAL FIX: BLOCK TENSORFLOW ---
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from transformers import pipeline
import yt_dlp
from fpdf import FPDF

# --- CONFIGURATION ---
MODEL_ID = "distil-whisper/distil-small.en"
BATCH_SIZE = 8

# Silence warnings
warnings.filterwarnings("ignore", message=".*chunk_length_s is very experimental.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    print("📂 Waiting for file selection...")
    file_path = filedialog.askopenfilename(
        title="Select your CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path

def load_model():
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        print("   ✅ Mac GPU (MPS) Detected & Enabled!")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        print("   ✅ NVIDIA GPU (CUDA) Detected!")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("   ⚠️ GPU not found. Using CPU.")

    print(f"⚡ Loading Model ('{MODEL_ID}')...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=torch_dtype,
        device=device
    )
    return pipe

def download_audio(youtube_url, output_filename="temp_audio"):
    print(f"⬇️  Downloading audio from: {youtube_url}...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_filename + ".mp3"

def transcribe_audio(pipe, audio_path):
    print("🎧 Transcribing...")
    result = pipe(
        audio_path, 
        chunk_length_s=30,
        batch_size=BATCH_SIZE, 
        return_timestamps=True
    )
    return result["text"]

def save_to_pdf(text, filename):
    if not filename.endswith(".pdf"):
        filename += ".pdf"
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, safe_text)
    
    pdf.output(filename)
    print(f"💾 Saved PDF: {filename}")

if __name__ == "__main__":
    csv_file = select_csv_file()
    
    if not csv_file:
        print("❌ No file selected. Exiting.")
        exit()
        
    print(f"✅ Selected: {csv_file}")
    ai_pipe = load_model()
    
    print("\n🚀 Starting Batch Processing...\n")

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Skip empty rows
            if not row: continue
            
            # --- 🧠 SMART COLUMN DETECTION ---
            # Try to find which column has the URL (starts with http)
            url = None
            name = None
            
            # Check Column 0 for URL
            if len(row) > 0 and row[0].strip().startswith("http"):
                url = row[0].strip()
                if len(row) > 1: name = row[1].strip()
            
            # Check Column 1 for URL (in case Col 0 is a number like 1, 2, 3)
            elif len(row) > 1 and row[1].strip().startswith("http"):
                url = row[1].strip()
                # If URL is in col 1, Name is likely in col 2
                if len(row) > 2: name = row[2].strip()
                # If there is no col 2, assume col 0 might be the name (unlikely but possible)
            
            # Fallback if no Name found but URL exists
            if url and not name:
                name = f"Transcript_{row[0]}"

            # Skip if we still couldn't find a URL
            if not url:
                print(f"⚠️ Skipping row (No URL found): {row}")
                continue

            print(f"\n🎥 Processing: {name}")
            temp_audio_name = f"temp_{name.replace(' ', '_').replace('/', '-')}"
            
            try:
                audio_path = download_audio(url, temp_audio_name)
                
                if os.path.exists(audio_path):
                    transcript = transcribe_audio(ai_pipe, audio_path)
                    save_to_pdf(transcript, name)
                    os.remove(audio_path)
                    print(f"✅ Finished: {name}")
                    
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

    print("\n✨ All Done!")