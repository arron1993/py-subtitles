import json

import whisperx
import gc

import srt

from datetime import timedelta

MODEL_CACHE_DIR = "./.model-cache"

class Video():
    def __init__(self, f_path):
        self.f_path = f_path

    def to_mp3(self):
        pass

    def get_audio_text(self):
        device = "cpu"
        audio_file = "out.mp3"
        batch_size = 16  # reduce if low on GPU mem
        compute_type = "float32"

        model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type, download_root=MODEL_CACHE_DIR
        )

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, language="en", batch_size=batch_size)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        return result
