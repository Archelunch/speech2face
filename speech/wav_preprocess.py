import glob
import os
from pathlib import Path
import numpy as np

from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm


encoder = VoiceEncoder()


def process_speaker(speaker: str, target_path: str) -> bool:
    speaker_name = speaker.split('/')[-1]
    speaker_folder = os.path.join(target_path, speaker_name)
    os.makedirs(speaker_folder)

    for audio_id, audio_filename in enumerate(glob.glob(os.path.join(speaker, "**/*.wav"), recursive=True)):
        fpath = Path(audio_filename)
        wav = preprocess_wav(fpath)
        embed = encoder.embed_utterance(wav)
        with open(os.path.join(speaker_folder, f"{audio_id}.npy"), "wb") as f:
            np.save(f, embed)

    return True


def process_dataset(path: str, target_path: str) -> bool:
    try:
        os.makedirs(target_path)
    except OSError as error:
        print("Folder already exists")
        return False

    speakers = [f for f in glob.glob(f"{path}/*")]

    for sp in tqdm(speakers):
        process_speaker(sp, target_path)

    print("Dataset is ready!")
    return True


if __name__ == "__main__":
    path = ""
    target_path = ""
    process_dataset(path, target_path)
