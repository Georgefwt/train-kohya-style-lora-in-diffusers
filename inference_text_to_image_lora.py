from diffusers import StableDiffusionPipeline
import torch
# use DDIM scheduler
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

model_path = "output/<output folder>"

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5", safety_checker=None)
pipe.load_lora_weights(model_path, weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.)
pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

generator = [torch.Generator(device="cuda").manual_seed(1024)]

prompt = "A close potrait of sks dog."
image = pipe(prompt,
             negative_prompt="...",
             height=512,
             width=512,
             num_inference_steps=30,
             guidance_scale=6.,
             generator = generator,
             ).images[0]
image.save("...")
