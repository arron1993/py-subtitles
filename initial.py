import json

import whisperx
import gc

import srt

from datetime import timedelta

MODEL_CACHE_DIR = "./.model-cache"


def main():
    with open("./result.json") as f:
        text_extract = json.load(f)

    subtitles = []
    for segment in text_extract:
        for word in segment["words"]:
            subtitles.append(
                srt.Subtitle(
                    start=timedelta(seconds=word["start"]),
                    end=timedelta(seconds=word["end"]),
                    content=word["word"],
                    index=len(subtitles),
                )
            )
    with open("./subtitles.srt", "w") as f:
        f.write(srt.compose(subtitles))
    return 0


def _get_text():
    device = "cpu"
    audio_file = "out.mp3"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float32"

    model = whisperx.load_model(
        "large-v2", device, compute_type=compute_type, download_root=MODEL_CACHE_DIR
    )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en", batch_size=batch_size)
    print(result["segments"])

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

    print(result["segments"])
    with open("result.json", "w") as f:
        json.dump(result["segments"], f)


if __name__ == "__main__":
    main()
