import argparse
import traceback
import sys
import io
from tqdm import tqdm 
from pathlib import Path

from src.config.configs import PipelineConfig, SequenceContext, BaseConfig, ALL, EVAL
from src.loader.dataset_pipeline import EgoHTGPipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human Terrain Data Generation.")
    parser.add_argument("--all", type=str, default=None, help="Process all sequences.")
    parser.add_argument("--sequence", type=str, default='sequence_01', help="Specific sequence to process.")
    parser.add_argument("--optimize", action= "store_true", help="Run optimization of processed data.")
    parser.add_argument("--rerun", action="store_true", help="Visualize generated sequence.")
    parser.add_argument("--web", action="store_true", default=False, help="Spawn rerun in browser.")
    parser.add_argument("--export", action="store_true", help="Export data into log file.")
    parser.add_argument("--duration", type=int, default=None, help="Length (s) of the sequence to output.")
    parser.add_argument("--retarget", action="store_true", help="Retarget to robot model.")
    parser.add_argument("--wandb", action="store_true", help="Use wandb to log optimization process.")
    parser.add_argument("--overwrite_gen", action= "store_true", help="Overwrite existing sequences.")
    parser.add_argument("--overwrite_opt", action= "store_true", help="Overwrite existing sequences.")
    parser.add_argument("--overwrite_retarget", action= "store_true", help="Overwrite existing sequences.")
    parser.add_argument("--eval", action="store_true", help="Evaluate generated sequences.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--ablation", action="store_true", help="Enable ablation study mode.")
    parser.add_argument("--multi_aria", action="store_true", help="Use multiple ARIA devices.")
    parser.add_argument("--period_retarget", type=int, nargs=2, default=[0, 350], help="Start and end time (s) for retargeting.")
    return parser.parse_args()

def init_error_log(ctx: BaseConfig, all_sequences: str) -> None:
    """Create/overwrite the error log file for this run."""
    err_path = ctx.error_logs
    err_path.parent.mkdir(parents=True, exist_ok=True)
    with err_path.open("w") as f:
        f.write(f"[INFO]: Error log for sequence {all_sequences}\n")


def log_error(ctx: SequenceContext, seq_name: str, exc: BaseException) -> None:
    """Append an error with traceback to the error log."""
    with ctx.error_logs.open("a") as f:
        f.write(f"\n=== Error for sequence: {seq_name} ===\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
        f.write("\n")

if __name__ == "__main__":
    #init
    root = Path(__file__).resolve().parent.parent
    
    args = get_args()
    pip_cfg = PipelineConfig.from_args(args, root)
    pipeline = EgoHTGPipeline(pip_cfg)
    

    if pip_cfg.all_sequences:
        init_error_log(BaseConfig(root), pip_cfg.all_sequences)  # Initialize error log at the start of the script
        sequences = ALL if pip_cfg.all_sequences == 'all' else EVAL if pip_cfg.all_sequences == 'eval' else []
    
        #use tqdm to track progress of multiple sequences
        sequences = tqdm(sequences, desc="Processing sequences")
        for seq in sequences:
            pipeline.pip_cfg.multi_aria = args.multi_aria 
            # Create SequenceContext
            seq_ctx = SequenceContext(name=seq, root=root)
            try:    
                pipeline.run(seq_ctx)
            except Exception as e:
                print(f"[ERROR]: Failed to run sequence {seq}: {e}")
                log_error(seq_ctx, seq, e)
    else: 
        seq_ctx = SequenceContext(name=pip_cfg.sequence_name, root=root)   
        pipeline.run(seq_ctx)
     
        
    try:        
        input("[INFO]: Exit server by pressing Enter!")
        
        print("[INFO]: Exiting server...")
    except KeyboardInterrupt:
        print("[WARNING]: Stopping server...")




    