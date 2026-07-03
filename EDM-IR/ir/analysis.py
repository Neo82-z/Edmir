import sys
from dataclasses import dataclass
from typing import Any




@dataclass(frozen=True)
class ExposurResult:
  phase_time_us: float
  critical_path_us: float
  total_comm_us: float
  exposed_comm_us:float
  total_data_movement_us:float
  exposed_data_movement_us:float
  ecr:float
  edmr:float
  critical_path:tuple[int, ...]


 def analyze_exposure(graph: EDMGraph) -> ExposureResult:
  ...
