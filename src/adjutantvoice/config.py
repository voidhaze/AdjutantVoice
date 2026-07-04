"""
AdjutantVoice configuration — all tuneable constants in one place.

Values can be overridden via environment variables (pydantic-settings).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Resolve the package assets directory regardless of working directory
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = _PACKAGE_DIR / "assets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AV_",        # e.g. AV_HOST=0.0.0.0
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8111
    mcp_port: int = 8222

    # Model
    model_id: str = "k2-fsa/OmniVoice"
    device: str = "cuda:0"
    dtype: str = "float16"              # "float16" | "bfloat16" | "float32"

    # Voice
    #
    # The voice clone is NOT bundled/checked into the repo — on a fresh
    # clone, `voice_clone_path` will not exist yet. `tts.load()` detects
    # this and falls back to `default_voice_instruct` (OmniVoice Voice
    # Design mode) instead. Run `av voice create-clone` once to generate
    # the default clone; after that it's picked up automatically.
    voice_clone_dir: Path = Path.home() / ".adjutantvoice" / "voices"
    default_voice_clone_name: str = "default"
    ref_audio_path: Path = ASSETS_DIR / "adjutant-terran-advisor-quotes.mp3"
    default_voice_instruct: str = "female"
    sample_rate: int = 24_000
    available_voices: list[str] = ["adjutant"]
    model_label: str = "omnivoice"

    # MCP output
    tts_output_dir: Path = Path("tts_output")

    @property
    def voice_clone_path(self) -> Path:
        """Resolved path to the default voice-clone pickle.

        Derived from `voice_clone_dir` + `default_voice_clone_name` so both
        can be overridden independently via AV_VOICE_CLONE_DIR /
        AV_DEFAULT_VOICE_CLONE_NAME.
        """
        return self.voice_clone_dir / f"{self.default_voice_clone_name}.pkl"


# Module-level singleton — import and use directly.
settings = Settings()
