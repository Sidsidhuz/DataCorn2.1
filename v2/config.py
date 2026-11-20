"""Explicit deployment modes. An unconfigured deployment exposes synthetic data only."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    mode: str = 'demo'
    data_dir: Path = Path(__file__).resolve().parents[1] / '.autoinsight'
    max_datasets: int = 20
    max_workers: int = 2
    job_timeout: int = 180
    retention_days: int = 30

    @classmethod
    def from_env(cls):
        mode = os.getenv('AUTOINSIGHT_MODE', 'demo').lower()
        if mode not in {'local', 'demo', 'private'}:
            raise ValueError('AUTOINSIGHT_MODE must be local, demo, or private.')
        root = Path(os.getenv('AUTOINSIGHT_DATA_DIR', str(cls.data_dir))).expanduser().resolve()
        return cls(mode=mode, data_dir=root)

    @property
    def uploads_allowed(self):
        return self.mode in {'local', 'private'}
