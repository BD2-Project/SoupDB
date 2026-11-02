"""SQL multimedia profiles: stable codebooks shared by indexed rows and queries.

Images use SIFT, audio segments use MFCC. The registered profile fixes the
codebook, dataset root and descriptor cache; no implicit training occurs here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from engine.common.errors import QueryExecutionError

_SEGMENT = re.compile(r"^(?P<path>.+)#segment=(?P<number>\d{1,10})$")


@dataclass(frozen=True, slots=True)
class MultimediaProfile:
    profile_id: str
    kind: Literal["image", "audio_segment"]
    root: Path
    codebook: Path
    cache: Path | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.strip():
            raise ValueError("multimedia profile_id is required")
        if self.kind not in ("image", "audio_segment"):
            raise ValueError("multimedia profile kind must be image or audio_segment")


class MultimediaQueryVectorizer:
    """Compute indexed and query vectors using exactly the same profile.

    Paths are dataset-root-relative, never absolute or outside the root.
    Audio references may carry ``#segment=000001``; without a suffix, the
    first 5-second audio segment is used. Empty/short audio is rejected.
    """

    def __init__(self, profiles: Mapping[str, MultimediaProfile]) -> None:
        self._profiles = dict(profiles)
        self._codebooks = {}
        for key, profile in self._profiles.items():
            if key != profile.profile_id:
                raise ValueError(f"profile registry key {key!r} must match profile_id")

    def profile(self, profile_id: str) -> MultimediaProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise QueryExecutionError(f"unknown multimedia profile {profile_id!r}") from None

    def codebook(self, profile_id: str):
        """Load and validate the profile codebook; return (centroids, idf)."""
        import numpy as np

        from engine.multimedia.codebook import load_codebook

        profile = self.profile(profile_id)
        if profile_id in self._codebooks:
            return self._codebooks[profile_id]
        try:
            centroids, idf = load_codebook(profile.codebook)
        except (OSError, KeyError, ValueError) as exc:
            raise QueryExecutionError(
                f"cannot load multimedia codebook {profile.codebook}: {exc}"
            ) from exc
        dimension = 128 if profile.kind == "image" else 20
        if (
            centroids.ndim != 2
            or centroids.shape[1] != dimension
            or centroids.shape[0] < 1
            or idf.shape != (len(centroids),)
            or not np.isfinite(centroids).all()
            or not np.isfinite(idf).all()
        ):
            raise QueryExecutionError(f"invalid codebook for profile {profile_id!r}")
        self._codebooks[profile_id] = (centroids, idf)
        return centroids, idf

    def dimension(self, profile_id: str) -> int:
        return int(len(self.codebook(profile_id)[0]))

    def vectorize(self, reference: str, *, profile_id: str) -> tuple[float, ...]:
        """Resolve a reference securely, then extract BoW TF-IDF and L2-normalize."""
        from engine.multimedia.codebook import assign_words, count_words, tfidf_vector

        profile = self.profile(profile_id)
        if not isinstance(reference, str) or not reference.strip():
            raise QueryExecutionError("multimedia reference must be a non-empty path")
        segment = 0
        rel = reference
        if profile.kind == "audio_segment":
            match = _SEGMENT.fullmatch(reference)
            if match:
                rel, segment = match["path"], int(match["number"])
            elif "#" in reference:
                raise QueryExecutionError("invalid audio #segment=NNNNNN suffix")
        elif "#segment=" in reference:
            raise QueryExecutionError("segment suffix is only valid for audio profiles")

        root = Path(profile.root).resolve()
        relative = Path(rel)
        if relative.is_absolute() or not relative.parts:
            raise QueryExecutionError("multimedia paths must be relative to the profile root")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise QueryExecutionError("multimedia file is missing or outside the profile root")

        centroids, idf = self.codebook(profile_id)
        if profile.kind == "image":
            from engine.multimedia.image_features import extract_sift, extract_sift_cached

            descriptors = (
                extract_sift_cached(path, profile.cache)
                if profile.cache is not None
                else extract_sift(path)
            )
        else:
            from engine.multimedia.audio_features import (
                extract_mfcc_cached,
                iter_audio_segments,
                mfcc_from_signal,
            )

            if profile.cache is not None:
                descriptors = extract_mfcc_cached(path, segment, profile.cache)
            else:
                segments = iter_audio_segments(path)
                try:
                    _, signal = next(part for part in segments if part[0] == segment)
                except StopIteration:
                    raise QueryExecutionError(
                        f"audio segment {segment} not found in {relative.as_posix()}"
                    ) from None
                descriptors = mfcc_from_signal(signal)
        histogram = count_words(assign_words(descriptors, centroids), len(centroids))
        result = tuple(float(v) for v in tfidf_vector(histogram, idf))
        if len(result) != len(centroids) or not all(isfinite(v) for v in result):
            raise QueryExecutionError("multimedia vector is non-finite or has wrong dimension")
        return result
