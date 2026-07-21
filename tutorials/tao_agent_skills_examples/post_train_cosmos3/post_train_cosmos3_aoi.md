# Industrial Defect Detection with Cosmos 3 Nano and TAO Agent Skills

This tutorial walks you through fine-tuning the NVIDIA Cosmos 3 Nano vision
language model for Automated Optical Inspection (AOI) to detect manufacturing defects across multiple industrial
object classes using the
[MVTec Anomaly Detection dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
[NVIDIA TAO Toolkit 7.0.1](https://docs.nvidia.com/tao/tao-toolkit/index.html),
and the [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank).

**TAO Agent Skills** are the key enabler here. Rather than hand-authoring
Docker commands, writing training configs, or debugging framework errors, you
install the TAO Skill Bank into your coding agent once and then describe what
you want in plain English. The `tao-finetune-cosmos-reason` skill packages
everything the agent needs — container definitions, config templates, dataset
format specifications, known error patterns, and evaluation scripts — so it
can reason through the full MLOps workflow on your behalf.

You drive the entire experiment with **four natural-language prompts**. The
agent handles dataset preparation, checkpoint conversion, container
orchestration, config generation, training, and evaluation automatically.

On the 5-class subset used in this tutorial, the workflow lifted accuracy from
**69.2%** (zero-shot) to **95.0%** after a single LoRA run — a gain of
**+25.8 percentage points** in approximately 32 minutes of wall-clock time on
4× RTX PRO 6000 Blackwell GPUs.

## Table of contents

1. [Install the TAO Skill Bank](#1-install-the-tao-skill-bank)
2. [Set up your environment](#2-set-up-your-environment)
3. [Download the MVTec dataset](#3-download-the-mvtec-dataset)
4. [Prompt 1: Prepare the dataset](#4-prompt-1-prepare-the-dataset)
5. [Prompt 2: Convert the Cosmos 3 Nano checkpoint](#5-prompt-2-convert-the-cosmos-3-nano-checkpoint)
6. [Prompt 3: Zero-shot baseline evaluation](#6-prompt-3-zero-shot-baseline-evaluation)
7. [Prompt 4: LoRA fine-tuning and evaluation](#7-prompt-4-lora-fine-tuning-and-evaluation)
8. [Results](#8-results)
9. [References](#references)

---

## 1. Install the TAO Skill Bank

Install the [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank) for
your coding agent.

For **Claude Code**, add the Skill Bank marketplace and install the plugin:

```text
/plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
/plugin install tao-skills@tao-skill-bank
```

For **Codex**, run:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-TAO/tao-skills-bank/main/scripts/install-codex-agents.sh | bash
```

Restart your agent after installation so the new skills load. The agent will
use the `tao-finetune-cosmos-reason` skill throughout this tutorial — it
packages the Cosmos-RL container definitions, configuration templates, dataset
format specifications, and evaluation scripts.

---

## 2. Set up your environment

Export your credentials in the shell **before** starting your agent so the
session inherits them:

```bash
export HF_TOKEN="hf_..."      # Hugging Face token — Cosmos 3 Nano requires access approval
export NGC_API_KEY="..."       # NGC API key — for pulling TAO containers
```

Accept the model agreement for
[`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) on Hugging
Face before continuing. Keep credential values in the environment; do not
paste them into the agent chat.

---

## 3. Download the MVTec dataset

Download the
[MVTec Anomaly Detection dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
from the MVTec website and extract it to a local directory. The dataset
contains 15 industrial object categories, each with normal training images and
defective test images.

```bash
mkdir mvtec_anomaly_detection
tar -xzf mvtec_anomaly_detection.tar.xz -C mvtec_anomaly_detection/
```

After extraction you will have a folder (`mvtec_anomaly_detection/` or similar)
with one subdirectory per category — `bottle/`, `cable/`, `capsule/`, etc. Note
the absolute path to this folder; you will pass it to the agent in Prompt 1.

This tutorial uses 5 categories chosen for their low zero-shot accuracy, which
makes the before/after improvement most dramatic:

| Category | Zero-shot accuracy | Why it is hard |
|----------|--------------------|----------------|
| capsule | 58% | Subtle surface scratches and cracks |
| cable | 63% | Complex multi-wire structure; bent or missing strands |
| screw | 75% | Small thread defects and scratches |
| zipper | 71% | Fine repeated texture; fabric defects |
| pill | 79% | Color contamination and chip defects |

---

## 4. Prompt 1: Prepare the dataset

The TAO cosmos-rl skill expects data in LLaVA conversation format — a JSON
file where each entry pairs an image with a human prompt and a model response.
For defect detection the prompt is a multiple-choice question with the object
class name injected, and the response is either `A) Not Defective` or
`B) Defective`.

Send this prompt to your coding agent:

```text
Reorganize the MVTec dataset in the current project folder at ./mvtec_anomaly_detection  into a VLM fine-tuning dataset
suitable for TAO cosmos-rl.

Use these 5 classes: capsule, cable, screw, pill, zipper.

For each class:
- Pool train/good and test/good as the Good class
- Pool all test/{defect_type} subdirectories as the Bad class
- Undersample Good to match Bad count (50/50 balance)
- Use 80 images per label for train and 12 per label for val

Format each annotation as a LLaVA-style JSON entry with a multiple-choice
prompt that injects the object class name. The answer choices should be:
  A) Not Defective
  B) Defective

Include an object_class field in each entry so eval scripts can report
per-class accuracy.

Save the dataset to ./vlm_detection_dataset with train/ and val/
splits, each containing an annotations.json and an images/ directory.
Print a summary table showing image counts per class and split when done.
```

The agent will write a dataset preparation script, run it, and confirm the
output structure. The result is 800 training samples and 120 val samples, with
all images balanced 50/50 per class.

### What the annotation looks like

```json
{
  "id": "capsule_train_Bad_0000",
  "image": "images/capsule/Bad/0000.png",
  "object_class": "capsule",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are a quality control inspector examining a capsule.\n\nQuestion: Based on the visual appearance of this capsule, what is its quality status?\n\nA) Not Defective\nB) Defective\n\nSelect the single best answer."
    },
    {
      "from": "gpt",
      "value": "B) Defective"
    }
  ]
}
```

---

## 5. Prompt 2: Convert the Cosmos 3 Nano checkpoint

The TAO cosmos-rl training container requires Cosmos 3 Nano weights in Qwen3-VL
safetensors format. The conversion downloads the base weights from Hugging Face
and merges them with the Cosmos3 vision tower into a single loadable checkpoint.

Send this prompt to your coding agent (in the same session):

```text
Download nvidia/Cosmos3-Nano from Hugging Face and convert it to the
Qwen3-VL safetensors format required by the TAO cosmos-rl container.

Save the converted checkpoint to ./models/Cosmos3-Nano-VLM .
Use the HF_TOKEN environment variable for authentication.

Confirm the conversion succeeded by showing the output files.
```

The agent will:
1. Pull the `tao-toolkit:7.0.1-pyt` container
2. Download the Cosmos 3 Nano weights (~33 GB) from Hugging Face
3. Merge the vision tower with the Qwen3-VL language model weights
4. Save 4 safetensor shards to `~/industrial_defect/ptm/Cosmos3-Nano-VLM/`

> **Note:** This is a one-time step. Once converted, the same checkpoint is
> reused for all future training and evaluation runs.

---

## 6. Prompt 3: Zero-shot baseline evaluation

Before any training, measure how well the base Cosmos 3 Nano performs on the
defect detection task with no fine-tuning. This establishes the baseline you
will compare against after LoRA training.

Send this prompt to your coding agent:

```text
Run zero-shot evaluation of Cosmos 3 Nano on the defect detection val set.

Base model: models/Cosmos3-Nano-VLM
Val set: vlm_detection_dataset/demo_vlm/val/annotations.json

Use all available GPUs in parallel to speed up inference — split the val set
into shards and run one shard per GPU. Merge the results and report
accuracy, precision, recall, and F1 per class, plus overall accuracy.

Do not load any LoRA adapter — this is the pure zero-shot baseline.
```

### Expected zero-shot results

| Class | Accuracy | F1 |
|-------|----------|-----|
| capsule | 58.3% | 0.308 |
| cable | 62.5% | 0.435 |
| zipper | 70.8% | 0.667 |
| screw | 75.0% | 0.571 |
| pill | 79.2% | 0.727 |
| **OVERALL** | **69.2%** | — |

The base model's precision is high (rarely calls a good part defective) but
recall is low — it misses many actual defects. Fine-tuning addresses this
directly.

---

## 7. Prompt 4: LoRA fine-tuning and evaluation

With the dataset prepared, the checkpoint converted, and the baseline recorded,
you are ready to fine-tune. Send this prompt to your coding agent:

```text
Fine-tune Cosmos 3 Nano with LoRA on the MVTec defect detection dataset using
TAO cosmos-rl. Then evaluate the best checkpoint and compare against the
zero-shot baseline.

Base model:    ./models/Cosmos3-Nano-VLM
Train data:    ./vlm_detection_dataset/train/annotations.json
Val data:      ./vlm_detection_dataset/val/annotations.json
Output:        ./output/

Training config:
- 4 GPUs, FSDP data-parallel sharding
- 4 epochs
- Learning rate: 2e-4 with cosine decay and 1 epoch warmup
- Batch size: 8 per GPU (effective batch 32)
- LoRA rank 16, alpha 32, dropout 0.05
- Target all attention and MLP projection layers
- Save a checkpoint and run validation after every epoch

After training, identify the best checkpoint by validation loss, then run
inference eval on the val set using all available GPUs and report per-class
accuracy, precision, recall, F1, and the delta vs zero-shot for each class.
```

The agent will:
1. Generate the `train_spec.toml` from the parameters above
2. Launch the `tao-toolkit:7.0.1-cosmos-rl` container on all 4 GPUs
3. Train for 4 epochs, saving a checkpoint after each
4. Select the best checkpoint (lowest val loss — typically epoch 3)
5. Run the per-class accuracy eval and print the comparison table

### Example LoRA configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2e-4 |
| LR schedule | Cosine decay, 1 epoch warmup |
| Batch size | 8 per GPU |
| Distributed strategy | FSDP, `dp_shard_size=4` |
| Epochs | 4 |
| Model sequence length | 8,192 |

### Val loss by epoch

| Epoch | Val Loss |
|-------|----------|
| 1 | 0.0641 |
| 2 | 0.0382 |
| **3** | **0.0323** ← best checkpoint |
| 4 | 0.0333 |

---

## 8. Results

### Accuracy: zero-shot vs LoRA fine-tuned

| Class | Zero-Shot | LoRA (ep3) | Delta |
|-------|-----------|------------|-------|
| capsule | 58.3% | 91.7% | **+33.3%** |
| cable | 62.5% | 95.8% | **+33.3%** |
| zipper | 70.8% | 100.0% | **+29.2%** |
| screw | 75.0% | 95.8% | **+20.8%** |
| pill | 79.2% | 91.7% | **+12.5%** |
| **OVERALL** | **69.2%** | **95.0%** | **+25.8%** |

### Timing (4× RTX PRO 6000 Blackwell)

| Phase | Wall-clock time |
|-------|----------------|
| Zero-shot eval (2 GPUs, 120 samples) | ~25 sec |
| LoRA training (4 GPUs, 4 epochs) | ~31 min |
| LoRA eval (2 GPUs, 120 samples) | ~45 sec |
| **Total** | **~32 min** |

Checkpoint conversion (`Prompt 2`) is a one-time step of ~5 minutes and does
not need to be repeated for future runs.

### Why this works

- **Object-aware prompting** injects the class name so the model applies
  class-specific visual priors — what a normal screw looks like is very
  different from a normal zipper.
- **Full-layer LoRA** (all attention + MLP projections) gives more expressive
  capacity than Q/V-only adapters, which matters for visual tasks where the
  defects are often subtle texture changes.
- **50/50 balanced training data** prevents the model from defaulting to
  predicting "Not Defective" for everything.

---

## References

- [MVTec Anomaly Detection Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [NVIDIA TAO Toolkit 7.0.1 Documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- [Cosmos 3 Nano on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano)
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank)
- [Post-Train NVIDIA Cosmos 3 in One Day Using Agent Skills](https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-in-one-day-using-agent-skills/)
- [Cosmos 3 Reasoner NIM on NGC](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-reasoner)
