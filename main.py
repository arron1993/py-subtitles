import json

import whisperx
import gc


def main():
    device = "cpu"
    audio_file = "out.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float32"

    model_cache_dir = "./cache/"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root="./cache")

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en", batch_size=batch_size)
    print(result["segments"])

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"])
    with open("result.json", "w") as f:
        json.dump(result['segments'], f)


if __name__ == '__main__':
    main()
