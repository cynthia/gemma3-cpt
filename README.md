# Gemma3 12B Multimodal CPT

WARNING: The training implementation in this repository was written as a thermal
stress workload, and is highly likely to produce garbage models. Do not use this
in production unless your hobby is lighting money on fire.

## Run Training

### Single Node (8 GPUs) with DeepSpeed
```bash
./run_deepspeed.sh
```

### Single Node (8 GPUs) with torchrun
```bash
./run_distributed.sh
```

### Single GPU
```bash
python train.py
```
