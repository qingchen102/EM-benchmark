"""多源多天线电磁信号仿真模块 (v2)。"""
from .factory import generate_signal_sample, generate_dataset, split_metadata
from .visualization import (
    spectrogram_image,
    music_spectrum,
    spatial_spectrum_image,
    save_sample_visualizations,
)

__all__ = [
    "generate_signal_sample",
    "generate_dataset",
    "split_metadata",
    "spectrogram_image",
    "music_spectrum",
    "spatial_spectrum_image",
    "save_sample_visualizations",
]
