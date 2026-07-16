# Post-train Cosmos 3 on WTS with TAO agent skills

This tutorial walks you through post-training the NVIDIA Cosmos 3 Nano vision
language model on a multiple-choice video question-answering task using a coding
agent and the [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank). It
is the hands-on companion to [Post-Train NVIDIA Cosmos 3 in One Day Using Agent
Skills](https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-in-one-day-using-agent-skills/).

You drive the entire workflow with two natural-language prompts:

1. **Prompt 1** establishes a zero-shot baseline, runs LoRA post-training, and
   evaluates the adapter.
2. **Prompt 2** runs a [TAO AutoML](https://docs.nvidia.com/tao/tao-toolkit/latest/text/automl/automl.html)
   sweep to tune hyperparameters and find a stronger configuration.

The coding agent, backed by the TAO skills, handles the framework, container,
config generation, data checks, error patching, and evaluation for you. On the
[Woven Traffic Safety (WTS) dataset](https://woven-visionai.github.io/wts-dataset-homepage/)
used in this example, the workflow lifted
exact-match validation accuracy from **54.41%** (zero-shot) to **87.14%** after
a single LoRA run, and to **93.35%** after the AutoML sweep. Your numbers will
depend on your dataset, checkpoint, and software versions.

## Table of contents

1. [Install the TAO Skill Bank](#1-install-the-tao-skill-bank)
2. [Set up your environment](#2-set-up-your-environment)
3. [Prepare your dataset](#3-prepare-your-dataset)
4. [Prompt 1: baseline, LoRA training, and evaluation](#4-prompt-1-baseline-lora-training-and-evaluation)
5. [Prompt 2: AutoML sweep](#5-prompt-2-automl-sweep)
6. [Results](#6-results)
7. [Deploy the post-trained model](#7-deploy-the-post-trained-model)
8. [References](#references)

## 1. Install the TAO Skill Bank

Install the [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank) for
your coding agent.

For **Codex**, run:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-TAO/tao-skills-bank/main/scripts/install-codex-agents.sh | bash
```

> **Note:** You must be signed in to GitHub in your terminal for the
> installation script to work (for example, via `gh auth login` or a configured
> SSH key/credential helper).

For **Claude Code**, add the Skill Bank marketplace and install the plugin:

```text
/plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
/plugin install tao-skills@tao-skill-bank
```

The installer registers the plugin and the TAO `AGENTS.md` identity. Restart
your agent after installation so the new plugin and instructions load. If you
prefer to drive each step yourself, follow the
[manual steps](https://github.com/NVIDIA-TAO/tao-skill-bank#manual-steps) in the
Skill Bank README.

For this workflow, the agent uses the `tao-finetune-cosmos-reason` skill. It
packages the Cosmos-RL action definitions, configuration templates, dataset
mapping, evaluation behavior, known error patterns, and AutoML parameters, so
the agent reasons through the workflow instead of relying on hand-copied
commands.

## 2. Set up your environment

TAO platform skills support
Slurm, Docker, Kubernetes, Brev, and more. You do not need to describe your
hardware up front: the agent uses the installed skills to discover the available
platform and GPUs, checks the environment, and confirms anything it needs before
launching. The example in this tutorial ran on a node of eight NVIDIA A100 80 GB
GPUs on OCI Slurm.

Export your credentials in the shell **before** starting your agent so the
session inherits them:

```bash
export HF_TOKEN="..."                 # Cosmos 3 checkpoint access
export NGC_KEY="..."                  # TAO container registry access
export AUTOML_LLM_API_KEY="..."       # Only for an LLM-guided AutoML proposer
```

Then accept the model agreement for
[`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) on Hugging
Face. Keep credential values in the environment; do not paste them into the
agent chat.

## 3. Prepare your dataset

Have your dataset ready before you start. This example uses the
[Woven Traffic Safety (WTS) dataset](https://woven-visionai.github.io/wts-dataset-homepage/),
a four-option video question-answering task, with train and validation roots
that already follow the LLaVA-style video format consumed by Cosmos:

```text
WTS_dataset/
├── wts_data_train/
│   ├── annotations.json
│   └── videos/
└── wts_data_val/
    ├── annotations.json
    └── videos/
```

Each data sample contains a video reference, a human turn with the `<video>`
token and the answer choices, and an assistant turn with the expected option
letter:

```json
{
  "id": "example-id",
  "video": "videos/example.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nWhat is the formation of the road?\nA: Single road (right curve)\nB: Single road (straight line)\nC: Intersection (without signal)\nD: Intersection (with signal)"
    },
    {
      "from": "gpt",
      "value": "D"
    }
  ]
}
```

You do not need to hand-validate the data. The agent checks the annotations,
media paths, split, and answer format as part of the workflow, and patches
recoverable issues automatically. In this example, the agent detected that the
selected runtime expected a video FPS field missing from the annotations and
generated the required patch on the fly, writing a derived annotation file
rather than modifying the source data.

## 4. Prompt 1: baseline, LoRA training, and evaluation

Start the post-training workflow with a prompt as short as this:

```text
Perform LoRA post-training of the Cosmos 3 model on the Woven Traffic Safety dataset.
Training data: /home/.../WTS_dataset/wts_data_train
Validation data: /home/.../WTS_dataset/wts_data_val
Base model on Hugging Face: nvidia/Cosmos3-Nano
Also perform a baseline evaluation first, to compare with the post-trained model.
```

From here, the agent selects the relevant TAO model and platform skills,
discovers your execution environment, and confirms any missing details with you
before launching. If you already know your platform, hardware, output location,
or metric, you can add them to the prompt to skip the back-and-forth:

```text
Use the TAO Skill Bank to post-train nvidia/Cosmos3-Nano with LoRA on the
Woven Traffic Safety multiple-choice video-QA task.

Platform: Slurm on OCI
Resources per job: 1 node, 8 NVIDIA A100 80 GB GPUs
Training root: <ABSOLUTE_TRAIN_ROOT>
Validation root: <ABSOLUTE_VALIDATION_ROOT>
Output root: <ABSOLUTE_OUTPUT_ROOT>

Evaluate the unmodified base model on all validation questions using normalized
exact option-letter accuracy. Then train a LoRA adapter, evaluate it on the same
validation set, and report the baseline and post-training results.
```

Once you confirm the launch plan, the agent runs the full sequence hands-free:

1. Selects `tao-finetune-cosmos-reason` and the platform skill for your
   environment.
2. Resolves the current container image and action contracts from the Skill
   Bank.
3. Validates the annotations, videos, credentials, resources, and output paths,
   fixing recoverable data or configuration issues along the way.
4. Generates the training and evaluation specs from the packaged templates.
5. Loads or converts the Cosmos 3 checkpoint if the selected image cannot
   consume the native format.
6. Evaluates the base model and saves per-record predictions plus aggregate
   accuracy.
7. Trains the LoRA adapter.
8. Evaluates the adapter using the same base model, prompt, validation records,
   decoding settings, and normalization as the baseline.
9. Reports a side-by-side comparison and keeps the resolved specs, logs,
   checkpoints, and metrics.

### Example LoRA configuration

For reference, the agent resolved these values for the single LoRA run in this
example. They document one run rather than universal defaults; the skill
validates the configuration against its schema, the dataset, and the GPU
topology it finds.

| Parameter | Value |
| --- | --- |
| Adapter targets | `q_proj`, `v_proj` |
| LoRA rank | 16 |
| LoRA alpha | 256 |
| LoRA dropout | 0.05 |
| Learning rate | `1e-5` |
| Batch size | 8 |
| Epochs | 1 |
| Model sequence length | 40,960 |
| Sampled video frames | 8 |
| Distributed strategy | FSDP, data-parallel shard size 8 |

### Result after prompt 1

| Model | Correct / total | Accuracy | Change from baseline |
| --- | ---: | ---: | ---: |
| Cosmos 3 Nano base | 1,456 / 2,676 | 54.41% | — |
| Base + LoRA | 2,332 / 2,676 | 87.14% | +32.73 percentage points |

In this example, the LoRA job took roughly 30 minutes on eight A100 80 GB GPUs,
raising accuracy by about 33 percentage points over the zero-shot baseline in a
single run.

## 5. Prompt 2: AutoML sweep

A single LoRA configuration already gives a large gain, but hyperparameters such
as learning rate, LoRA rank, dropout, and batch size all affect final accuracy.
Instead of tuning them by hand, ask the agent to run a TAO AutoML sweep. Send the
second prompt in the same chat so it reuses the validated data, checkpoint,
metric, platform, and outputs:

```text
Run an AutoML sweep to improve the LoRA result. Let TAO choose suitable search strategies
and tune the important training hyperparameters. Optimize validation accuracy and summarize
the best models.
```

If you want tighter control over the search space and reporting, be more
explicit:

```text
Continue from the LoRA run. Use TAO AutoML on the same platform and the same
train/validation split. Tune LoRA rank, LoRA alpha, LoRA dropout, learning rate,
weight decay, warmup epochs, batch size, and number of epochs. Keep the base
checkpoint, exact-match evaluator, decoding, and answer normalization fixed
across trials.

Use normalized exact option-letter accuracy as the objective to maximize. Show
the proposed samplers, parameter ranges, number of trials, concurrency, GPU
resources, estimated per-trial and total cost, output root, and first candidate
configuration before asking me to approve the sweep. After approval, evaluate
every completed adapter, rank trials by exact-match accuracy, and also report
validation loss.
```

[TAO AutoML](https://docs.nvidia.com/tao/tao-toolkit/latest/text/automl/automl.html)
supports several search strategies, including Bayesian optimization,
Hyperband, Bayesian optimization and Hyperband (BOHB), batch-first Bayesian
optimization (BFBO), and LLM-guided search (which proposes hyperparameters using
an LLM Brain like NVIDIA NIM, OpenAI, or other OpenAI-compatible endpoint and requires
`NVIDIA_API_KEY` or `AUTOML_LLM_API_KEY`).

In this example, the agent launched 43 trials across multiple strategies:

| Sampler | Trials |
| --- | ---: |
| Bayesian | 10 |
| Batch-first Bayesian optimization (BFBO) | 10 |
| Bayesian optimization and Hyperband (BOHB) | 3 |
| LLM-guided Bayesian | 10 |
| LLM-guided BFBO | 10 |

The sweep tuned eight hyperparameters: LoRA rank, LoRA alpha, LoRA dropout,
learning rate, weight decay, warmup epochs, batch size, and number of epochs.
The number of epochs was itself part of the search space (trials used one to
three epochs); the best-accuracy trial happened to converge at a single epoch.

### Optimize for the right metric

The AutoML controller can optimize validation loss or task accuracy, and the two
do not always agree. In this example, the trial with the lowest validation loss
was not the one with the highest accuracy:

| Selection | Sampler | Validation loss | Exact-match accuracy |
| --- | --- | ---: | ---: |
| Highest accuracy | Bayesian | 0.0969 | **93.35%** |
| Lowest loss | LLM-guided Bayesian | **0.0956** | 93.31% |

If accuracy is your goal, use the explicit prompt above and configure the AutoML
evaluator to maximize exact-match accuracy directly, and have the agent report
both metrics so any disagreement is visible.

### Best trial

The best-accuracy trial used a Bayesian sampler with this configuration:

| Parameter | Value |
| --- | --- |
| Sampler | Bayesian |
| LoRA rank | 64 |
| LoRA alpha | 2,048 |
| LoRA dropout | 0.0377 |
| Learning rate | `2.59e-4` |
| Learning-rate schedule | Linear decay |
| Batch size | 32 |
| Weight decay | 0.0145 |
| Epochs | 1 |

Best accuracy by sampler:

| Sampler | Best exact-match accuracy |
| --- | ---: |
| Bayesian | **93.35%** |
| LLM-guided Bayesian | 93.31% |
| BOHB | 93.27% |
| BFBO | 92.75% |
| LLM-guided BFBO | 92.64% |

The full 43-trial sweep took about 19.5 hours of wall-clock time with trials
distributed across multiple nodes (roughly 170 GPU-hours). Individual trials ran
in about 30 minutes each. AutoML cost depends on queueing, concurrency, failed
trials, model cache state, and validation duration.

## 6. Results

Across the two prompts, accuracy on the shared 2,676-question validation set
improved as follows:

| Variant | Validation accuracy | Change from base |
| --- | ---: | ---: |
| Base model (zero-shot) | 54.41% | — |
| LoRA | 87.14% | +32.73 pp |
| AutoML LoRA | **93.35%** | +38.94 pp |

All three entries use the same validation set and the same normalized
exact-letter scoring, so the comparison is apples-to-apples. The agent keeps the
resolved specs, logs, adapter checkpoints, per-record predictions, and metrics
for each run, so you can audit any number rather than trusting a bare
percentage.

### LoRA versus full SFT

The same AutoML workflow was also run with full-parameter supervised
fine-tuning (SFT) instead of LoRA. The best SFT trial reached the same 93.35%
validation accuracy as the best AutoML LoRA trial, but full SFT updates every
weight and cost roughly 7x more GPU-hours per trial. For this task, LoRA matched
full fine-tuning accuracy at a fraction of the compute, which is why this
tutorial uses it as the default post-training method.

## 7. Deploy the post-trained model

The [Cosmos 3 Reasoner NIM](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-reasoner)
is the fastest path to a production-grade, OpenAI-compatible endpoint for the
post-trained model. It ships as a prebuilt, optimized container, so you skip
manual vLLM dependencies and CUDA-pairing configuration, and it serves text
outputs from text, image, and video inputs.

You can ask the coding agent to deploy it for you:

```text
Locally deploy Cosmos 3 Nano Reasoner NIM using the post-trained checkpoint in <checkpoints path>. Run NVIDIA/Docker preflight, use a LoRA-capable profile when available or the fused merged checkpoint otherwise.
```

An important difference from LLaMA-style VLMs: those NIMs can serve a base model
plus standalone LoRA adapter directories, but the Cosmos 3 Reasoner NIM expects
a **merged checkpoint** (base weights fused with the LoRA adapter) rather than a
bare adapter. The agent handles the merge and provides the container with a
checkpoint it can load.

Once the container is up, run the health checks:

```bash
curl -sS http://localhost:8000/v1/health/ready
curl -sS http://localhost:8000/v1/models | jq
```

### Deploy manually

If you prefer to deploy by hand, pull the container from the
[Cosmos 3 Reasoner NIM catalog page](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-reasoner)
and follow the video walkthrough
[How to Run NVIDIA Cosmos 3 Reasoner NIM for Video Reasoning](https://www.youtube.com/watch?v=Z8BkL4A96LI)
for step-by-step serving instructions.

The [NIM LoRA fine-tuning guide](https://docs.nvidia.com/nim/vision-language-models/latest/finetune-lora.html)
documents adapter serving (mounting a directory via `NIM_PEFT_SOURCE`, selecting
a `-feat_lora` profile, and static vs. dynamic loading), but that adapter-based
path applies to NIMs that support standalone LoRA adapters, such as LLaMA. For
the Cosmos 3 Reasoner NIM, supply the merged checkpoint as described above.

## References

- [NVIDIA technical blog: Post-Train NVIDIA Cosmos 3 in One Day Using Agent Skills](https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-in-one-day-using-agent-skills/)
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank)
- [Cosmos 3 Nano on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano)
- [Woven Traffic Safety dataset](https://woven-visionai.github.io/wts-dataset-homepage/)
- [TAO AutoML documentation](https://docs.nvidia.com/tao/tao-toolkit/latest/text/automl/automl.html)
- [Cosmos 3 Reasoner NIM on NGC](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-reasoner)
- [Video: How to Run NVIDIA Cosmos 3 Reasoner NIM for Video Reasoning](https://www.youtube.com/watch?v=Z8BkL4A96LI)
- [NIM for VLMs: Fine-Tuning with LoRA](https://docs.nvidia.com/nim/vision-language-models/latest/finetune-lora.html)
