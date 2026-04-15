from animRL.utils.task_registry import task_registry

from .mimic.mimic_task import MimicTask
from .mimic.mimic_hw_task import MimicHWTask
from ..cfg.mimic.walk_config import WalkCfg, WalkTrainCfg
from ..cfg.mimic.walk_hw_config import WalkHWCfg, WalkHWTrainCfg

task_registry.register("walk", MimicTask, WalkCfg(), WalkTrainCfg())
task_registry.register("walk-hw", MimicHWTask, WalkHWCfg(), WalkHWTrainCfg())
