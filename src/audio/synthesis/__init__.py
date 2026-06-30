"""Text-to-speech backend abstractions."""

from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult
from src.audio.synthesis.dataset import AudioDatasetSynthesisError, synthesize_text_dataset_audio
from src.audio.synthesis.metadata import (
    AudioMetadataError,
    DEFAULT_AUDIO_METADATA_PATH,
    audio_sample_from_synthesis_result,
    write_audio_metadata,
)
from src.audio.synthesis.piper import PiperTTSBackend, PiperTTSBackendError
from src.audio.synthesis.tts_backend import (
    TTSAdapter,
    TTSAdapterConfigError,
    create_tts_adapter_from_config,
    create_tts_backend_from_config,
)

__all__ = [
    "AudioDatasetSynthesisError",
    "AudioMetadataError",
    "DEFAULT_AUDIO_METADATA_PATH",
    "PiperTTSBackend",
    "PiperTTSBackendError",
    "TTSAdapter",
    "TTSAdapterConfigError",
    "TTSBackend",
    "TTSSynthesisRequest",
    "TTSSynthesisResult",
    "audio_sample_from_synthesis_result",
    "create_tts_adapter_from_config",
    "create_tts_backend_from_config",
    "synthesize_text_dataset_audio",
    "write_audio_metadata",
]
