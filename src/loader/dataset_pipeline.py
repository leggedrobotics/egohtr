from src.loader.scene_loader import SceneLoader
from src.loader.egomc_loader import EgoMoCapLoader
from src.loader.aria_loader import AriaLoader
from src.loader.exomc_loader import ExoMoCapLoader
    
from src.config.configs import PipelineConfig, SequenceContext
from src.visualizer.data_visualizer import EgoHTGVisualizer
from src.loader.data_generator import EgoHTGGenerator
from src.retargeter.motion_retargeter import EgoHTGRetargeter
from src.optimizer.data_optimizer import EgoHTGOptimizer
from src.evaluator.data_evaluator import EgoHTGEvaluator
from src.body_model.smplx import SMPLXHandler
    
from src.loader.helper.helper import load_config


class EgoHTGPipeline():
    """
    Pipeline to manage the EgoHumanTerrainGeneration Dataset. This pipeline handles the full process of data generation, optimization, evaluation, retargeting, 
    and visualization for each sequence in the dataset. It loads the necessary configurations, initializes the appropriate data loaders and body models.
    """
    
    def __init__(self,
        cfg: PipelineConfig
        ):
        self.pip_cfg = cfg
        self.sensor_cfg = load_config(self.pip_cfg.sensor_config_path)
        
        self.generator = EgoHTGGenerator(self.pip_cfg, self.sensor_cfg)
        self.retargeter = EgoHTGRetargeter(self.pip_cfg)
        self.optimizer = EgoHTGOptimizer(self.pip_cfg)
        
        if self.pip_cfg.eval:
            self.evaluator = EgoHTGEvaluator(self.pip_cfg)
        if self.pip_cfg.rerun:
            self.visualizer = EgoHTGVisualizer(self.pip_cfg)
        
    def run(self, ctx: SequenceContext):
        """Run the full pipeline for a specific sequence."""
        seq_cfg = load_config(ctx.config_yaml)
        
        loader = self._init_loaders(ctx, seq_cfg)
        if loader == {}:
            raise ValueError(f"Failed to initialize loaders for sequence {ctx.name}.")
        
        # Initialize body models
        self.get_active_body_models(ctx, seq_cfg)
        
        # go throough pipeline steps
        if not ctx.gen_seq.exists() or self.pip_cfg.overwrite_gen:
            self.generate_sequence(loader, ctx, seq_cfg)
        if self.pip_cfg.optimize and (not ctx.opt_seq.exists() or self.pip_cfg.overwrite_opt):
            self.optimize_sequence(ctx, seq_cfg)
        if self.pip_cfg.eval and ctx.eval_data_exists:
            self.evaluate_sequence(ctx, seq_cfg)
        if self.pip_cfg.retarget:
            self.retarget_sequence(ctx, seq_cfg)
        if self.pip_cfg.rerun:
            self.visualize_sequence(loader, ctx, seq_cfg)
    

    def get_active_body_models(self, ctx: SequenceContext, seq_cfg: dict) -> dict: 
        active_sensors = []
        if ctx.egomc_exists: active_sensors.append("egomc")
        if ctx.exomc_exists: active_sensors.append("exomc")

        if not active_sensors:
            raise ValueError("At least one of EgoMoCap or ExoMoCap data must be present.")

        self.body_models = {}

        for sensor in active_sensors:
            handler = SMPLXHandler(self.pip_cfg, self.retargeter, sensor_type=sensor)
            handler.initialize(seq_cfg["smplx_config"])
            self.body_models[sensor] = handler

    def generate_sequence(self, loader: dict, ctx: SequenceContext, seq_cfg: dict):
        """Generate data for a specific sequence."""
        print("\n###### Generate sequence:", ctx.name, "######")
        self.generator.run(loader, ctx, seq_cfg["generator_config"], self.body_models, self.optimizer)
        
    def optimize_sequence(self, ctx: SequenceContext, seq_cfg: dict):
        """Optimize data for a specific sequence."""
        print("\n###### Optimize sequence:", ctx.name, "######")
        self.optimizer.run(ctx, seq_cfg["optimizer_config"]["ik_config"], self.body_models["egomc"])
    
    def evaluate_sequence(self, ctx: SequenceContext, seq_cfg: dict):
        """Evaluate data for a specific sequence."""
        print("\n###### Evaluate sequence:", ctx.name, "######")
        self.evaluator.run(ctx, seq_cfg["evaluator_config"])
    
    def retarget_sequence(self, ctx: SequenceContext, seq_cfg: dict):
        """Retarget data for a specific sequence."""
        print("\n###### Retarget sequence:", ctx.name, "######")
        self.retargeter.run(ctx, seq_cfg["retargeter_config"])

    def visualize_sequence(self, loader: dict, ctx: SequenceContext, seq_cfg: dict):
        """Visualize data for a specific sequence."""
        print("\n###### Visualize sequence:", ctx.name, "######")
        self.visualizer.run(loader, ctx, seq_cfg, seq_cfg["smplx_config"]["gender"])
        
        
    def _init_loaders(self, ctx: SequenceContext, seq_cfg: dict) -> dict:
        """Initialize data loaders based on the configuration."""
        
        # aria loaders
        if self.pip_cfg.multi_aria:
            if not ctx.aria_mutli_exists:
                print("[INFO] Multi-ARIA data not found in the sequence context. Change to single ARIA mode.")
                self.pip_cfg.multi_aria = False
            aria_loader = AriaLoader(ctx.aria_multi_subject, self.sensor_cfg['aria'], self.pip_cfg, seq_cfg)
            aria_obs_loader = AriaLoader(ctx.aria_multi_observer, self.sensor_cfg['aria'], self.pip_cfg, seq_cfg)
        else:
            aria_loader = AriaLoader(ctx.aria_subject, self.sensor_cfg['aria'], self.pip_cfg, seq_cfg)
            aria_obs_loader = None

        # optional loaders
        egomc_loader = EgoMoCapLoader(ctx,  self.sensor_cfg['egomc'], self.pip_cfg, seq_cfg) if ctx.egomc_exists else None
        scene_loader = SceneLoader(ctx, self.sensor_cfg['scanner'], self.pip_cfg, seq_cfg) if ctx.scene_exists else None
        exomc_loader = ExoMoCapLoader(ctx, self.sensor_cfg['exomc'], self.pip_cfg, seq_cfg) if ctx.exomc_exists else None

        return {
            "aria": aria_loader,
            "aria_obs": aria_obs_loader,
            "egomc": egomc_loader,
            "scene": scene_loader,
            "exomc": exomc_loader
        }