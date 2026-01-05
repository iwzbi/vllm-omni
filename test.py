a = {'stage_id': 0, 'stage_type': 'diffusion', 'runtime': {'process': True, 'devices': '0', 'max_batch_size': 1}, 'engine_args': {'model': '/model/stable-diffusion-3.5-medium/', 'dtype': 'float16', 'cache_backend': 'none', 'cache_config': None, 'model_stage': 'diffusion'}, 'final_output': True, 'final_output_type': 'image'}
import yaml

with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(a, f, sort_keys=False, allow_unicode=True)
