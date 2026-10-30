import argparse
import os
from dataclasses import dataclass
from pathlib import Path

EVAL_XHALL = [
    "exo-standing_s1_xhall_01",
    "exo-standing_s4_xhall_01",
    "exo-standing_s5_xhall_01",
    "exo-standing_s6_xhall_01",
    "exo-walking_s4_xhall_01",
    "exo-walking_s5_xhall_01",
    "exo-high-object_s4_xhall_01",
    "exo-high-object_s5_xhall_01",
    "exo-high-object_s6_xhall_01",
    "exo-cliff-stones_s4_xhall_01",
    "exo-cliff-stones_s5_xhall_01",
    "exo-parkour_s1_xhall_01",
    "exo-parkour_s4_xhall_01",
    "exo-parkour_s5_xhall_01",
    "exo-parkour_s6_xhall_01",
]

EXOMC_XHALL = [
    "exomc-parkour_xhall_01",
    "exomc-parkour_xhall_02",
    "exomc-parkour_xhall_03",
    "exomc-parkour_xhall_04",
    "exomc-parkour_xhall_05",
    "exomc-parkour_xhall_06",
]

LEE = [
    "double-stairs_lee_01",
    "double-stairs_lee_02",
]

XHALL1 = [
    "stepping-stones_xhall_01",
    "high-object_xhall_01",
    "double-high-object_xhall_01",
    "narrow-path_xhall_01",
    "cliff-path_xhall_01",
    "crawling_xhall_01",
]

XHALL2 = [
    "exomc-parkour_xhall_01",
    "exomc-parkour_xhall_02",
    "exomc-parkour_xhall_03",
    "exomc-parkour_xhall_04",
    "exomc-parkour_xhall_05",
    "exomc-parkour_xhall_06",
]

XHALL3 = [
    "exo-standing_s1_xhall_01",
    "exo-standing_s6_xhall_01",
    "exo-walking_s4_xhall_01",
    "exo-walking_s5_xhall_01",
    "exo-high-object_s4_xhall_01",
    "exo-high-object_s5_xhall_01",
    "exo-cliff-stones_s4_xhall_01",
    "exo-cliff-stones_s5_xhall_01",
    "exo-parkour_s4_xhall_01",
]

GYM = [
    "climbing_s3_gym_01",
    "crawling_s3_gym_01",
    "double-high-object_s2_gym_01",
    "overhang-object_s2_gym_03",
    "parkour_s1_gym_02",
    "parkour_s1_gym_03",
    "parkour_s3_gym_01",
    "walking_s2_gym_01",
]

GYM2 = [
    "parkour_s1_gym_04",
    "parkour_s1_gym_05",
    "parkour_s1_gym_06",
    "parkour_s1_gym_07",
]

BZB = [
    "crawling_s1_bzb_02",
    "crawling_s1_bzb_03",       
    "cliff_s4_bzb_01",
    "cliff-jump_s4_bzb_01",
    "cliff-jump_s4_bzb_02",
    "window_s4_bzb_01",
    "rescue_s4_bzb_01",
    "rocky-path_s7_bzb_01",
    "rocky-path_s7_bzb_02",
    "window-rocky_s7_bzb_01",
    "window-rocky_s7_bzb_02",
    "walking_s7_bzb_01",
    "walking_rocky_s7_bzb_01",
    "walking_rocky_s7_bzb_02",
    "cliff-jump_s7_bzb_01",
    "cliff-jump_s8_bzb_01",
    "window_s8_bzb_01",
    "rescue_s8_bzb_01",
    "rocky-path_s8_bzb_01",
]

EVAL = XHALL2 + XHALL3
ITW = BZB + GYM + GYM2 + XHALL1 + LEE
ALL = BZB + XHALL3 + XHALL2 + LEE + XHALL1 + GYM2 + GYM


@dataclass
class BaseConfig:
    """Base configuration class with common properties."""
    root: Path
    
    @property
    def sensor_config_path(self): return self.root / "src" / "config" / "sensor_params.yaml"
    @property
    def data_path(self): return self.root / "data" / "sequences"
    @property
    def scene_path(self): return self.root / "data" / "scenes"
    @property
    def log_path(self): return self.root / "logs" / "sequences"
    @property
    def error_logs(self): return self.root/ "logs" / "error" / "error_logs.txt"
    @property
    def stdout_logs(self): return self.root / "logs" / "error" / "stdout_logs.txt"
    @property
    def models_path(self): return self.root / "data" / "models"
    @property
    def smplx_models(self): return self.models_path / "smplx_models" / "smplx"
    @property
    def robot_models(self): return self.models_path / "robot_models"

@dataclass 
class PipelineConfig(BaseConfig):
    """Configuration for the data pipeline."""
    sequence_name: str = "sequence_01"
    all_sequences: bool = False
    optimize: bool = False
    rerun: bool = False
    web: bool = False
    export: bool = False
    duration: int | None = None
    retarget: bool = False
    wandb: bool = False
    eval: bool = False
    root: Path = Path(".")
    debug: bool = False
    overwrite_gen: bool = False
    overwrite_opt: bool = False
    overwrite_retarget: bool = False
    ablation: bool = False
    multi_aria: bool = False
    period_retarget: list[int] = None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace, root: Path):
        """Converts argparse Namespace to our structured Dataclass."""
        return cls(
            all_sequences=args.all,
            sequence_name=args.sequence,
            optimize=args.optimize,
            rerun=args.rerun,
            web=args.web,
            export=args.export,
            duration=args.duration,
            retarget=args.retarget,
            wandb=args.wandb,
            eval=args.eval,
            debug=args.debug,
            overwrite_gen=args.overwrite_gen,
            overwrite_opt=args.overwrite_opt,
            overwrite_retarget=args.overwrite_retarget,
            ablation=args.ablation,
            multi_aria=args.multi_aria,
            period_retarget=args.period_retarget,
            root=root,
        )


@dataclass
class AriaObserverContext:
    """A container for 2nd perspective ARIA paths."""
    
    name: str
    aria_dir: Path

    @property
    def role(self): return "obs"
    @property
    def aria_vrs(self): return self.aria_dir / f"{self.name}_2nd.vrs"
    @property
    def aria_json(self): return self.aria_dir / f"{self.name}_2nd.vrs.json"
    @property
    def aria_mps_dir(self): return self.aria_dir / "multi_slam" / "1" 
    @property
    def aria_slam_dir(self): return self.aria_mps_dir / "slam"
    @property
    def aria_closed_loop_traj(self): return self.aria_slam_dir / "closed_loop_trajectory.csv"
    @property
    def global_pcd(self): return self.aria_slam_dir / "semidense_points.csv.gz"
    @property
    def aria_online_calib(self): return self.aria_slam_dir / "online_calibration.jsonl"
    @property
    def aria_audio_dir(self): return self.aria_dir / "audio"
    @property
    def aria_imgs_dir(self): return self.aria_dir / "imgs"
  

@dataclass
class AriaSubjectMultiContext:
    """A container for 1st perspective ARIA paths with multiple ARIAs."""
    
    name: str
    aria_dir: Path

    @property
    def role(self): return "sub"
    @property
    def aria_vrs(self): return self.aria_dir / f"{self.name}.vrs"
    @property
    def aria_json(self): return self.aria_dir / f"{self.name}.vrs.json"
    @property
    def aria_mps_dir(self): return self.aria_dir / "multi_slam" / "0" 
    @property
    def aria_slam_dir(self): return self.aria_mps_dir / "slam"
    @property
    def aria_closed_loop_traj(self): return self.aria_slam_dir / "closed_loop_trajectory.csv"
    @property
    def global_pcd(self): return self.aria_slam_dir / "semidense_points.csv.gz"
    @property
    def aria_online_calib(self): return self.aria_slam_dir / "online_calibration.jsonl"
    @property
    def aria_audio_dir(self): return self.aria_dir / "audio"
    @property
    def aria_imgs_dir(self): return self.aria_dir / "imgs"


@dataclass
class AriaSubjectSingleContext:
    """A container for 1st perspective ARIA paths with a single ARIA."""
    name: str
    aria_dir: Path

    @property
    def role(self): return "sub"
    @property
    def aria_vrs(self): return self.aria_dir / f"{self.name}.vrs"
    @property
    def aria_json(self): return self.aria_dir / f"{self.name}.vrs.json"
    @property
    def aria_mps_dir(self): return self.aria_dir / f"mps_{self.name}_vrs"
    @property
    def aria_slam_dir(self): return self.aria_mps_dir / "slam"
    @property
    def aria_closed_loop_traj(self): return self.aria_slam_dir / "closed_loop_trajectory.csv"
    @property
    def global_pcd(self): return self.aria_slam_dir / "semidense_points.csv.gz"
    @property
    def aria_online_calib(self): return self.aria_slam_dir / "online_calibration.jsonl" 
    @property
    def aria_audio_dir(self): return self.aria_dir / "audio"
    @property
    def aria_imgs_dir(self): return self.aria_dir / "imgs"
    


@dataclass
class SequenceContext(BaseConfig):
    """A central container for all paths related to a specific sequence."""
    name: str
    
    # General
    @property
    def seq_dir(self): return self.data_path / self.name
    @property
    def log_dir(self): return self.log_path / self.name
    @property
    def rerun(self): return self.log_dir / f"{self.name}.rrd"
    @property
    def config_yaml(self): return self.seq_dir / "config.yaml"
    @property
    def gen_seq(self): return self.seq_dir / "full_sequence.npz"
    @property
    def gen_seq_exo(self): return self.seq_dir / "full_exomc_sequence.npz"
    @property
    def opt_seq(self): return self.seq_dir / "full_opt_sequence.npz"
    @property
    def retarget_seq(self): return self.seq_dir / "full_retarget_sequence.npz"

    # Aria
    @property
    def aria_dir(self): return self.seq_dir / "aria"
    @property
    def aria_multi_subject(self): return AriaSubjectMultiContext(name=self.name, aria_dir=self.aria_dir)
    @property
    def aria_multi_observer(self): return AriaObserverContext(name=self.name, aria_dir=self.aria_dir)
    @property
    def aria_subject(self): return AriaSubjectSingleContext(name=self.name, aria_dir=self.aria_dir)
    @property
    def aria_mutli_exists(self):
        return self.aria_multi_subject.aria_closed_loop_traj.exists() and self.aria_multi_observer.aria_closed_loop_traj.exists()
    @property
    def aria_vrs_exists(self):
        return self.aria_subject.aria_vrs.exists()

    # EgoMocap
    @property
    def egomc_dir(self): return self.seq_dir / "egomc"
    @property
    def egomc_bvh(self): return self.egomc_dir / f"{self.name}_hik.bvh"
    @property
    def egomc_csv(self): return self.egomc_dir / f"{self.name}.csv"
    @property
    def egomc_exists(self):
        return self.egomc_bvh.exists() and self.egomc_csv.exists()
    
    # ExoMocap
    @property
    def exomc_dir(self): return self.seq_dir / "exomc"
    @property
    def exomc_c3d(self): return self.exomc_dir / f"{self.name}.c3d"
    @property
    def exomc_json(self): return self.exomc_dir / f"{self.name}.json"
    @property
    def exomc_exists(self):
        return self.exomc_json.exists()

    # Scanner 
    @property
    def scene_dir(self): return self.seq_dir / "scene"
    @property
    def scene_obj(self): return self.scene_dir / f"{self.name}.obj"
    @property
    def scene_mtl(self): return self.scene_dir / f"{self.name}.mtl"
    @property
    def scene_pcd(self): return self.scene_dir / f"{self.name}.e57"
    @property
    def scene_exists(self):
        return any([
            self.scene_pcd.exists(),
            self.scene_obj.exists()
        ])

    #Retargeter
    @property
    def retarget_dir(self): return self.seq_dir / "retarget"
    @property
    def scene_decomposed(self): return self.retarget_dir / "decomposed"
    @property
    def combined_mesh(self): return self.retarget_dir / "multi_boxes.obj"
    @property
    def relative_scene_decomposed(self): return Path(f"../../../sequences/{self.name}/retarget/decomposed")
    @property
    def box_assets_xml(self): return self.retarget_dir / "box_assets.xml"
    @property
    def box_body_xml(self): return self.retarget_dir / "box_body.xml"

    #Evaluator
    @property
    def eval_dir(self): return self.log_dir / "evaluation"
    @property
    def metrics_file(self): return self.eval_dir / "metrics.txt"
    @property
    def plots_dir(self): return self.eval_dir / "plots"
    @property
    def josh_data(self): 
        josh_dir = self.log_dir / "josh"
        #sort by alphanumeric order to get the latest one if there are multiple
        scene = sorted(josh_dir.glob("scene_*.pkl"))
        # optional external folder with JOSH runs (<EGOHTG_JOSH_DIR>/<sequence>/**/josh/scene_*.pkl)
        if os.environ.get("EGOHTG_JOSH_DIR"):
            scene.extend(sorted((Path(os.environ["EGOHTG_JOSH_DIR"]) / self.name).rglob("**/josh/scene_*.pkl")))
        return scene #scene[0] if scene else josh_dir / "scene.pkl"
    @property
    def josh_data_exists(self):
        return self.josh_data[0].exists() if self.josh_data else False
    @property
    def egoallo_data(self):
        return self.log_dir / "egoallo" / "joint_positions.npz"
    @property
    def egoallo_data_exists(self):
        return self.egoallo_data.exists()
    @property
    def eval_data_exists(self):
        return self.egomc_exists and (self.exomc_exists or self.josh_data_exists or self.egoallo_data_exists)


    # Visualizer
    @property
    def blueprint_path(self): return self.root / "data" / "blueprints" / "egohtg data visualizer.rbl"


@dataclass
class PreparationConfig(BaseConfig):
    """Configuration specific to the data preparation stage."""
    
    @property
    def description_file(self): return self.data_path / "description.md"
    @property
    def example_seq_dir(self): return self.data_path / "00_example_seq"
