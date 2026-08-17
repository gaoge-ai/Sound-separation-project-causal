import csv
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, get_worker_info

from rir_augment import RIRAugmenter
from utils.utils_library_gpu import RI_split, compute_fft, enframe


TARGET_SAMPLE_RATE = 16000
FRAME_SIZE = 512
FRAME_SHIFT = 256
NUM_SOURCES = 3
CATEGORIES = ("speech", "concert", "bird")
PATH_COLUMNS = ("audio_path", "path", "filename")


class OnlineMixDataset(Dataset):
    """Build speech/concert/bird mixtures from raw single-category audio files."""

    def __init__(
        self,
        source_csv,
        audio_root=None,
        samples_per_epoch=20000,
        num_sources_choices=(2, 3),
        num_sources_probs=None,
        snr_min=-3.0,
        snr_max=3.0,
        target_sample_rate=TARGET_SAMPLE_RATE,
        target_duration=10.0,
        seed=42,
        deterministic=False,
        rir_root=None,
        rir_prob=0.0,
        rir_room_probs=None,
    ):
        self.source_csv = Path(source_csv)
        self.audio_root = Path(audio_root) if audio_root else None
        self.samples_per_epoch = int(samples_per_epoch)
        self.num_sources_choices = tuple(int(x) for x in num_sources_choices)
        self.num_sources_probs = (
            None if num_sources_probs is None else tuple(float(x) for x in num_sources_probs)
        )
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.target_sample_rate = int(target_sample_rate)
        self.target_duration = float(target_duration)
        self.target_num_samples = int(round(self.target_sample_rate * self.target_duration))
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.window = torch.hamming_window(FRAME_SIZE, periodic=False, dtype=torch.float32)
        self.rir_augmenter = RIRAugmenter(
            rir_root=rir_root,
            target_sample_rate=self.target_sample_rate,
            rir_prob=rir_prob,
            room_probs=rir_room_probs,
        )

        self._validate_config()
        self.category_to_records = self._load_category_records()

    def _validate_config(self):
        if self.samples_per_epoch <= 0:
            raise ValueError("--samples_per_epoch must be greater than 0")
        if self.target_num_samples <= 0:
            raise ValueError("--target_duration must produce at least one sample")
        if not self.num_sources_choices:
            raise ValueError("--online_num_sources must contain at least one value")
        invalid_num_sources = [x for x in self.num_sources_choices if x < 1 or x > len(CATEGORIES)]
        if invalid_num_sources:
            raise ValueError(
                f"--online_num_sources values must be between 1 and {len(CATEGORIES)}: "
                f"{invalid_num_sources}"
            )
        if self.num_sources_probs is not None:
            if len(self.num_sources_probs) != len(self.num_sources_choices):
                raise ValueError(
                    "--online_num_sources_probs must have the same length as "
                    "--online_num_sources"
                )
            negative_probs = [x for x in self.num_sources_probs if x < 0]
            if negative_probs:
                raise ValueError(
                    f"--online_num_sources_probs values must be non-negative: {negative_probs}"
                )
            if sum(self.num_sources_probs) <= 0:
                raise ValueError("--online_num_sources_probs must sum to a positive value")
        if self.snr_min > self.snr_max:
            raise ValueError("--snr_min must be <= --snr_max")

    def _load_category_records(self):
        if not self.source_csv.exists():
            raise FileNotFoundError(f"Source CSV does not exist: {self.source_csv}")

        category_to_records = {category: [] for category in CATEGORIES}
        with self.source_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            if reader.fieldnames is None:
                raise ValueError(f"Source CSV is empty: {self.source_csv}")

            fieldnames = set(reader.fieldnames)
            if "category" not in fieldnames:
                raise ValueError(f"Source CSV must contain a 'category' column: {self.source_csv}")

            path_column = next((column for column in PATH_COLUMNS if column in fieldnames), None)
            if path_column is None:
                raise ValueError(
                    f"Source CSV must contain one of path columns {PATH_COLUMNS}: {self.source_csv}"
                )

            for row_number, row in enumerate(reader, start=2):
                category = str(row.get("category", "")).strip().lower()
                if category not in category_to_records:
                    raise ValueError(
                        f"Unsupported category '{category}' at {self.source_csv}:{row_number}; "
                        f"expected one of {CATEGORIES}"
                    )

                raw_path = str(row.get(path_column, "")).strip()
                if not raw_path:
                    raise ValueError(f"Empty audio path at {self.source_csv}:{row_number}")
                category_to_records[category].append(
                    {
                        "path": self._resolve_audio_path(raw_path),
                        "category": category,
                        "source": str(row.get("source", "")).strip(),
                    }
                )

        empty_categories = [
            category for category, records in category_to_records.items() if not records
        ]
        if empty_categories:
            raise ValueError(
                f"Source CSV must contain at least one sample for every category; "
                f"empty categories: {empty_categories}"
            )

        return category_to_records

    def _resolve_audio_path(self, raw_path):
        path = Path(raw_path)
        if not path.is_absolute() and self.audio_root is not None:
            path = self.audio_root / path

        if path.suffix.lower() == ".mp4":
            wav_path = path.with_suffix(".wav")
            if wav_path.exists() or not path.exists():
                path = wav_path

        return str(path)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = self._make_rng(idx)
        selected_categories = self._sample_categories(rng)
        snrs = [0.0] + [rng.uniform(self.snr_min, self.snr_max) for _ in range(len(selected_categories) - 1)]

        audios = []
        for category in selected_categories:
            record = rng.choice(self.category_to_records[category])
            audio = self._load_wav(record["path"], rng)
            audio = self.rir_augmenter.apply(
                audio=audio,
                category=record["category"],
                source=record["source"],
                rng=rng,
                target_num_samples=self.target_num_samples,
            )
            audios.append(audio)

        mixed_wav, scaled_sources = self._mix_audios(audios, snrs)
        category_targets = {
            category: torch.zeros_like(mixed_wav)
            for category in CATEGORIES
        }
        for category, source in zip(selected_categories, scaled_sources):
            category_targets[category] = category_targets[category] + source

        X1 = self._wav_to_ri(mixed_wav)
        Y_targets = [self._wav_to_ri(category_targets[category]) for category in CATEGORIES]
        R_targets = [category_targets[category].float() for category in CATEGORIES]

        return (X1, *Y_targets, *R_targets)

    def _make_rng(self, idx):
        if self.deterministic:
            return random.Random(self.seed + int(idx))

        worker_info = get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else random.randrange(0, 2**32)
        return random.Random(worker_seed + int(idx))

    def _sample_categories(self, rng):
        if self.num_sources_probs is None:
            num_sources = rng.choice(self.num_sources_choices)
        else:
            num_sources = rng.choices(
                self.num_sources_choices,
                weights=self.num_sources_probs,
                k=1,
            )[0]
        return rng.sample(list(CATEGORIES), k=num_sources)

    def _load_wav(self, path, rng):
        wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sample_rate != self.target_sample_rate:
            gcd = math.gcd(int(sample_rate), int(self.target_sample_rate))
            wav = resample_poly(
                wav,
                up=self.target_sample_rate // gcd,
                down=sample_rate // gcd,
            ).astype(np.float32, copy=False)

        if wav.size > self.target_num_samples:
            max_start = wav.size - self.target_num_samples
            start = rng.randint(0, max_start)
            wav = wav[start:start + self.target_num_samples]
        elif wav.size < self.target_num_samples:
            wav = np.pad(wav, (0, self.target_num_samples - wav.size), mode="constant")

        return torch.from_numpy(wav.astype(np.float32, copy=False))

    @staticmethod
    def _mix_audios(audios, snrs):
        target = audios[0]
        target_energy = torch.sum(target ** 2)
        mixed = target.clone()
        scaled_audios = [target]

        for i in range(1, len(audios)):
            noise = audios[i]
            noise_energy = torch.sum(noise ** 2)
            snr_linear = 10 ** (float(snrs[i]) / 10)
            scale = torch.sqrt((target_energy / snr_linear) / (noise_energy + 1e-8))
            scaled_noise = noise * scale
            mixed = mixed + scaled_noise
            scaled_audios.append(scaled_noise)

        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            scale = 0.9 / max_value
            mixed = mixed * scale
            scaled_audios = [audio * scale for audio in scaled_audios]

        return mixed.float(), [audio.float() for audio in scaled_audios]

    def _wav_to_ri(self, wav):
        frames = enframe(wav.unsqueeze(0), FRAME_SIZE, FRAME_SHIFT, self.window)
        freq = compute_fft(frames, FRAME_SIZE)
        return RI_split(freq, freq.shape[1]).squeeze(0).float()
