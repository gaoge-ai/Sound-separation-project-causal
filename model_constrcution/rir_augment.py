import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly


ROOMS = ("small", "medium", "large")
DEFAULT_ELIGIBLE_CONCERT_SOURCES = ("song_cut_202604221637",)


class RIRAugmenter:
    def __init__(
        self,
        rir_root=None,
        target_sample_rate=16000,
        rir_prob=0.0,
        room_probs=None,
        eligible_concert_sources=DEFAULT_ELIGIBLE_CONCERT_SOURCES,
    ):
        self.rir_root = Path(rir_root) if rir_root else None
        self.target_sample_rate = int(target_sample_rate)
        self.rir_prob = float(rir_prob)
        self.room_probs = self._parse_room_probs(room_probs)
        self.eligible_concert_sources = {
            str(source).strip() for source in eligible_concert_sources
        }
        self.enabled = self.rir_root is not None and self.rir_prob > 0
        self.rir_paths_by_room = {room: [] for room in ROOMS}

        if self.enabled:
            self.rir_paths_by_room = self._load_rir_paths()

    @staticmethod
    def _parse_room_probs(room_probs):
        if room_probs is None:
            return {room: 1.0 for room in ROOMS}

        if isinstance(room_probs, dict):
            probs = {room: float(room_probs.get(room, 0.0)) for room in ROOMS}
        else:
            values = [float(value) for value in room_probs]
            if len(values) != len(ROOMS):
                raise ValueError(
                    f"rir_room_probs must contain {len(ROOMS)} values "
                    f"for {ROOMS}, got {len(values)}"
                )
            probs = dict(zip(ROOMS, values))

        if any(value < 0 for value in probs.values()):
            raise ValueError("rir_room_probs values must be non-negative")
        if sum(probs.values()) <= 0:
            raise ValueError("rir_room_probs must sum to a positive value")
        return probs

    def _load_rir_paths(self):
        if not self.rir_root.exists():
            raise FileNotFoundError(f"RIR root does not exist: {self.rir_root}")

        rir_paths_by_room = {room: [] for room in ROOMS}
        for path in self.rir_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".wav", ".flac"):
                continue

            lower_path = str(path).lower()
            for room in ROOMS:
                if room in lower_path:
                    rir_paths_by_room[room].append(str(path))
                    break

        missing_rooms = [
            room
            for room, prob in self.room_probs.items()
            if prob > 0 and not rir_paths_by_room[room]
        ]
        if missing_rooms:
            raise ValueError(
                f"No RIR files found for rooms with non-zero probability: {missing_rooms}"
            )

        return rir_paths_by_room

    def should_apply(self, category, source, rng):
        if not self.enabled:
            return False

        category = str(category).strip().lower()
        source = str(source).strip()

        if category == "speech":
            eligible = True
        elif category == "concert":
            eligible = source in self.eligible_concert_sources
        else:
            eligible = False

        return eligible and rng.random() < self.rir_prob

    def apply(self, audio, category, source, rng, target_num_samples):
        if not self.should_apply(category, source, rng):
            return audio

        was_tensor = torch.is_tensor(audio)
        device = audio.device if was_tensor else None
        dtype = audio.dtype if was_tensor else None
        audio_np = self._to_numpy(audio)

        rir_path = self.sample_rir_path(rng)
        rir = self.load_rir(rir_path)
        augmented = self.convolve_and_trim(audio_np, rir, int(target_num_samples))

        if was_tensor:
            return torch.from_numpy(augmented).to(device=device, dtype=dtype)
        return augmented

    @staticmethod
    def _to_numpy(audio):
        if torch.is_tensor(audio):
            return audio.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(audio, dtype=np.float32)

    def sample_rir_path(self, rng):
        rooms = list(ROOMS)
        weights = [self.room_probs[room] for room in rooms]
        room = rng.choices(rooms, weights=weights, k=1)[0]
        return rng.choice(self.rir_paths_by_room[room])

    def load_rir(self, path):
        rir, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if rir.ndim > 1:
            rir = np.mean(rir, axis=1)
        if sample_rate != self.target_sample_rate:
            gcd = math.gcd(int(sample_rate), self.target_sample_rate)
            rir = resample_poly(
                rir,
                up=self.target_sample_rate // gcd,
                down=sample_rate // gcd,
            ).astype(np.float32, copy=False)
        return rir.astype(np.float32, copy=False)

    @staticmethod
    def convolve_and_trim(audio, rir, target_num_samples):
        reverbed = fftconvolve(audio, rir, mode="full")
        reverbed = reverbed[:target_num_samples]
        if reverbed.size < target_num_samples:
            reverbed = np.pad(
                reverbed,
                (0, target_num_samples - reverbed.size),
                mode="constant",
            )
        return reverbed.astype(np.float32, copy=False)
