import torch

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

ulysses_degree = 2

if __name__ == "__main__":
    generator = torch.Generator("cuda").manual_seed(42)
    model_name = "/model/stable-diffusion-3.5-medium/"
    m = Omni(
        model=model_name,
        # parallel_config=DiffusionParallelConfig(ulysses_degree=2),
        # output_type="latent",
        dtype=torch.half,
        stage_configs_path="/workspace/code/my_omni/config.yaml"
    )

    # high resolution may cause OOM on L4
    height = 256
    width = 256
    images = m.generate(
        "a cat wearing sunglasses, cyberpunk style",
        height=height,
        width=width,
        dtype=torch.half,
        num_inference_steps=1,
        guidance_scale=0,
        generator=generator,
        num_outputs_per_prompt=1,
    )
    list(images)
