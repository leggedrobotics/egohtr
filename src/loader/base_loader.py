import numpy as np

from abc import ABC, abstractmethod
from src.config.configs import SequenceContext, PipelineConfig

class DataLoader(ABC):
    """Abstract base class for all data loaders."""
    def __init__(self, ctx: SequenceContext, sensor_cfg: dict, pip_cfg: PipelineConfig, seq_cfg: dict):
        self.ctx = ctx
        self.sensor_cfg = sensor_cfg
        self.pip_cfg = pip_cfg
        self.seq_cfg = seq_cfg["generator_config"]
    
    @abstractmethod
    def load_data(self):
        """Load data from the specified sequence context."""
        pass
    
    @abstractmethod
    def process(self):
        """Process the loaded data."""
        pass

    def set_nframes_by_duration(self, duration_s: int):
        """Set number of frames based on synced duration with all devices."""
        self.duration = duration_s
        self.nframes = int(duration_s * self.sensor_cfg["fps"])
        self.end_frame_idx = self.start_frame_idx + self.nframes

    def get_synced_timestamp(self):
        """ Get timestamp array for timesynced ExoMoCap data """
        fps = self.sensor_cfg["fps"]
        return np.arange(self.nframes) / fps
    