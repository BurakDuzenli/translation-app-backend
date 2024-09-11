from flask import Flask, request, jsonify, send_file
from yt_dlp import YoutubeDL
import faster_whisper
import os
import tempfile
import subprocess
from pathlib import Path
import requests
from urllib.parse import urlsplit
import shutil

app = Flask(__name__)

# Global variable to store the loaded model
loaded_model = None
current_model_size = None

def seconds_to_time_format(s):
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/process_video", methods=['POST'])
def process_video():
    video_type = request.form.get('type')
    language = request.form.get('language', 'auto')
    initial_prompt = request.form.get('initial_prompt', '')
    word_level_timestamps = request.form.get('word_level_timestamps', 'false').lower() == 'true'
    vad_filter = request.form.get('vad_filter', 'true').lower() == 'true'
    vad_filter_min_silence_duration_ms = int(request.form.get('vad_filter_min_silence_duration_ms', 50))
    text_only = request.form.get('text_only', 'false').lower() == 'true'
    model_size = request.form.get('model_size', 'base')

    video_path_local_list = []

    if video_type == "Youtube video or playlist":
        url = request.form.get('url')
        video_path_local_list = process_youtube(url)
    elif video_type == "Google Drive":
        file = request.files.get('file')
        if file:
            video_path_local_list = process_google_drive(file)
    elif video_type == "Direct download":
        ddl_url = request.form.get('ddl_url')
        video_path_local_list = process_direct_download(ddl_url)
    else:
        return jsonify({"error": "Unsupported input type"}), 400

    results = []
    for video_path_local in video_path_local_list:
        wav_path = convert_to_wav(video_path_local)
        result = transcribe_audio(wav_path, language, initial_prompt, word_level_timestamps, vad_filter, vad_filter_min_silence_duration_ms, text_only, model_size)
        results.append(result)
        os.remove(wav_path)
        os.remove(video_path_local)

    return jsonify({"results": results})

@app.route("/download_transcription", methods=['POST'])
def download_transcription():
    data = request.json
    text_only = data.get('text_only', False)
    transcription = data.get('transcription', [])

    ext_name = '.txt' if text_only else ".srt"
    output_file_name = f"transcription{ext_name}"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=ext_name) as temp_file:
        sentence_idx = 1
        for item in transcription:
            if not text_only:
                temp_file.write(f"{sentence_idx}\n")
                temp_file.write(f"{item['start']} --> {item['end']}\n")
                temp_file.write(f"{item['text']}\n\n")
            else:
                temp_file.write(f"{item['text']}\n")
            sentence_idx += 1

    return send_file(temp_file.name, as_attachment=True, download_name=output_file_name)

def process_youtube(url):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
    return [Path(f"{info['id']}.wav")]

def process_google_drive(file):
    filename = file.filename
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(file_path)
    return [Path(file_path)]

def process_direct_download(ddl_url):
    response = requests.get(ddl_url)
    if response.status_code == 200:
        filename = urlsplit(ddl_url).path.split("/")[-1]
        file_path = os.path.join(tempfile.gettempdir(), filename)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return [Path(file_path)]
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def convert_to_wav(video_path):
    valid_suffixes = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".3gp", ".mpeg"]
    if video_path.suffix.lower() not in valid_suffixes:
        return str(video_path)

    wav_path = video_path.with_suffix(".wav")
    subprocess.run([
        "ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(wav_path)
    ])
    return str(wav_path)

def load_model(model_size):
    global loaded_model, current_model_size
    if loaded_model is None or current_model_size != model_size:
        loaded_model = faster_whisper.load_model(model_size)
        current_model_size = model_size
    return loaded_model

def transcribe_audio(audio_file, language, initial_prompt, word_level_timestamps, vad_filter, vad_filter_min_silence_duration_ms, text_only, model_size):
    model = load_model(model_size)
    
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language=None if language == "auto" else language,
        initial_prompt=initial_prompt,
        word_timestamps=word_level_timestamps,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms)
    )

    output = {
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "transcription": []
    }

    for segment in segments:
        if word_level_timestamps:
            for word in segment.words:
                ts_start = seconds_to_time_format(word.start)
                ts_end = seconds_to_time_format(word.end)
                output["transcription"].append({
                    "start": ts_start,
                    "end": ts_end,
                    "text": word.word
                })
        else:
            ts_start = seconds_to_time_format(segment.start)
            ts_end = seconds_to_time_format(segment.end)
            output["transcription"].append({
                "start": ts_start,
                "end": ts_end,
                "text": segment.text.strip()
            })

    return output

if __name__ == "__main__":
    app.run(debug=True)