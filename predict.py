import os
import shutil
import torch
from diffusers import DDIMScheduler, PNDMScheduler
from cog import BasePredictor, Input, Path
from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler

from model import Model

MODEL_CACHE = "diffusers-cache"
#MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_VAE = "stabilityai/sd-vae-ft-ema"

MODELS = ["runwayml/stable-diffusion-v1-5", "prompthero/openjourney-v4", "admruul/anything-v3.0", "nitrosocke/mo-di-diffusion"]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.model = Model(device='cuda', dtype=torch.float16)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        model_name: str = Input(
            description="Model name on huggingface",
            default="runwayml/stable-diffusion-v1-5"
        ),
        prompt: str = Input(
            description="Input prompt.",
            default="a cat",
        ),
        negative_prompt: str = Input(
            description="Negative prompt.",
            default="",
        ),
        seed: int = Input(
            description=" Leave blank to randomize the seed.",
            default=None,
        ),
        chunk_size: int = Input(
            description="Chunk size: Number of frames processed at once. Reduce for lower memory usage.",
            ge=1,
            le=10,
            default=8,
        ),
        motion_field_strength_x: int = Input(
            description="Global Translation $\delta_{x}$",
            ge=-20,
            le=20,
            default=12,
        ),
        motion_field_strength_y: int = Input(
            description="Global Translation $\delta_{y}$",
            ge=-20,
            le=20,
            default=12,
        ),
        t0: int = Input(
            description="Timestep t0: Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
            ge=1,
            le=50,
            default=44,
        ),
        t1: int = Input(
            description="Timestep t1: Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
            ge=1,
            le=50,
            default=47,
        ),
        merging_ratio: float = Input(
            description="Ratio of how many tokens are merged. The higher the more compression (less memory and faster inference).",
            ge=0,
            le=0.9,
            default=0,
        ),
        resolution: int = Input(
            description="Resolution of the video (square)",
            default=512,
        ),
        video_length: int = Input(
            description="Number of frames in the video",
            default=8,
        ),
        fps: int = Input(
            description="Frame rate for the video.", default=15, ge=5, le=60
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            print("Setting Random seed.")
            seed = int.from_bytes(os.urandom(2), "big")

        outdir = "cog_out/out"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
        path = os.path.join(outdir, 'out.mp4')

        self.model.process_text2video(
                        prompt=prompt, 
                        n_prompt=negative_prompt,
                        seed=seed, 
                        path=path, 
                        model_name=model_name, 
                        motion_field_strength_x=motion_field_strength_x,
                        motion_field_strength_y=motion_field_strength_y,
                        t0=t0,
                        t1=t1,
                        chunk_size=chunk_size,
                        video_length=video_length,
                        watermark=None,                          
                        merging_ratio=merging_ratio,
                        resolution=resolution,
                        fps=fps,
                        use_cf_attn=True,
                        use_motion_field=True,
                        smooth_bg=False,
                        smooth_bg_strength=0.4)

        video_path = f"cog_out/out/out.mp4"

        return Path(video_path)