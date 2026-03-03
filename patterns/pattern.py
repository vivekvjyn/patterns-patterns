import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import compiam
from compiam.melody.pitch_extraction import Melodia
import librosa
from swift_f0 import SwiftF0
import matplotlib.pyplot as plt
import numpy as np
import crepe
from bs_roformer import MODEL_REGISTRY, DEFAULT_MODEL, get_model_from_config

class Pattern:
    def __init__(self, srcsep_algorithm='ht-demucs', pe_algorithm='crepe'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Source separator setup
        self.srcsep_algorithm = srcsep_algorithm
        if self.srcsep_algorithm == 'ht-demucs':
            self.source_separator = get_model(name="htdemucs")
            self.source_separator.eval()
            self.source_separator.to(self.device)
        elif self.srcsep_algorithm == 'bs-roformer':
            entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
            config = ConfigDict(yaml.safe_load(open(f"models/{entry.slug}/{entry.config}")))
            self.source_separator = get_model_from_config("bs_roformer", config)
            state_dict = torch.load(f"models/{entry.slug}/{entry.checkpoint}", map_location="cpu")
            self.source_separator.load_state_dict(state_dict)

        # Pitch extractor setup
        self.pe_algorithm = pe_algorithm
        if self.pe_algorithm == 'melodia':
            self.pitch_extractor = Melodia()
        elif self.pe_algorithm == 'swiftf0':
            self.pitch_extractor = SwiftF0()
        elif self.pe_algorithm == 'crepe':
            self.pitch_extractor = crepe
        elif self.pe_algorithm == 'ftanet-carnatic':
            self.pitch_extractor = compiam.load_model("melody:ftanet-carnatic")
        else:
            raise ValueError(f"Unsupported pitch extraction algorithm: {self.pe_algorithm}.")


    def __call__(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        vocals = self._separate_sources(audio, sample_rate)

        pitch = self._extract_pitch(vocals, sample_rate)

    def _separate_sources(self, audio, sample_rate):
        audio = audio.unsqueeze(0).to(self.device)
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        if self.srcsep_algorithm == 'ht-demucs':
            with torch.no_grad():
                sources = apply_model(self.source_separator, audio, shifts=1, split=True)
                vocals = sources[0, 3].cpu().numpy()
                vocals = np.mean(vocals, axis=0)
        elif self.srcsep_algorithm == 'bs-roformer':
            with torch.no_grad():
                sources = self.source_separator(audio)
                vocals = sources[0, self.source_separator.sources.index("vocals")]
                vocals = vocals.cpu().numpy()
                vocals = np.mean(vocals, axis=0)

        torchaudio.save("vocals.wav", torch.from_numpy(vocals).unsqueeze(0), sample_rate)

        return vocals

    def _extract_pitch(self, audio, sample_rate):
        if self.pe_algorithm == 'melodia':
            pitch_track = self.pitch_extractor.extract(audio, sample_rate)
            times = pitch_track[:, 0]
            pitch = pitch_track[:, 1]
        elif self.pe_algorithm == 'swiftf0':
            pitch_result = self.pitch_extractor.detect_from_array(audio, sample_rate)
            pitch = pitch_result.pitch_hz
            pitch[~pitch_result.voicing] = 0.0
            times = pitch_result.timestamps
        elif self.pe_algorithm == 'crepe':
            times, pitch, confidence, _ = self.pitch_extractor.predict(audio, sample_rate, viterbi=True)
            pitch[confidence < 0.5] = 0.0
        elif self.pe_algorithm == 'ftanet-carnatic':
            pitch_track = self.pitch_extractor.predict(audio, sample_rate)
            times = pitch_track[:, 0]
            pitch = pitch_track[:, 1]

        # plot pitch on spectrogram
        plt.figure(figsize=(20, 6))
        # compute Spectrogram
        S = librosa.stft(audio, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.pcolormesh(librosa.times_like(S_db, sr=sample_rate, hop_length=512), librosa.fft_frequencies(sr=sample_rate, n_fft=2048), S_db, shading='gouraud')
        plt.plot(times, pitch, color='r', label='Pitch')
        plt.legend()
        plt.title('Spectrogram with Pitch')
        #plt.colorbar(format='%+2.0f dB')
        plt.ylim(0, 600)
        plt.tight_layout()
        plt.savefig('spectrogram_with_pitch.png')
        plt.close()

        return pitch
