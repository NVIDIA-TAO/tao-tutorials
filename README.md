# TAO Tutorials

<!-- vscode-markdown-toc -->
* [Requirements](#Requirements)
	* [Recommended Hardware requirements](#RecommendedHardwarerequirements)
	* [Software requirements](#Softwarerequirements)
* [Package Content](#PackageContent)
	* [File Hierarchy](#FileHierarchy)
* [How to run TAO?](#HowtorunTAO)
* [Important Links](#ImportantLinks)
* [Blogs](#Blogs)
* [License](#License)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

NVIDIA TAO, is a python based AI toolkit that is built on TensorFlow and PyTorch for computer vision applications. It simplifies and accelerates the model training process by abstracting away the complexity of AI models and the underlying deep learning framework.
You can use the power of transfer learning to fine-tune NVIDIA pretrained models with your own data and optimize the model for inference throughput — all without the need for AI expertise or large training datasets.

[TAO quick start video](https://www.nvidia.com/en-us/on-demand/session/other2022-tao/).

## <a name='Requirements'></a>Requirements

### <a name='RecommendedHardwarerequirements'></a>Recommended Hardware requirements

The following system configuration is recommended to achieve reasonable training performance with TAO and supported models provided:

* 16 GB system RAM
* 16 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

TAO is supported on discrete GPUs, such as H100, A100, A40, A30, A2, A16, A100x, A30x, Titan-RTX and Quadro-RTX.

> Note: TAO is not supported on GPU's before the Volta generation

### <a name='Softwarerequirements'></a>Software requirements

| **Software**                     | **Version** |
| :--- | :--- |
| Ubuntu LTS                       | 22.04       |
| python                           | >=3.10.x    |
| docker-ce                        | >19.03.5    |
| docker-API                       | 1.40        |
| `nvidia-container-toolkit`       | >1.3.0-1    |
| nvidia-container-runtime         | 3.4.0-1     |
| nvidia-docker2                   | 2.5.0-1     |
| nvidia-driver                    | >550.xx     |
| python-pip                       | >21.06      |

## <a name='PackageContent'></a>Package Content

Download the TAO tutorial package which contains startup scripts, Jupyter notebooks and config files. <br>

    git clone https://github.com/NVIDIA/tao_tutorials.git
    cd ./tao_tutorials


### <a name='FileHierarchy'></a>File Hierarchy

    setup
        |--> quickstart_launcher.sh
    tutorials
        |--> tao_api_starter_kit
            |--> api
                |--> automl
                |--> end2end
                |--> dataset_prepare
            |--> client
                |--> automl
                |--> end2end
                |--> dataset_prepare
        |--> tao_launcher_starter_kit
            |--> dino
            |--> deformable_detr
            |--> classification_pyt
            |--> grounding_dino
            |--> ocdnet
            |-->  ...
        |--> tao_data_services
            |--> data
            |-->  ...
        |--> tao_agent_skills_examples
            |--> post_train_cosmos3
                |--> post_train_cosmos3_lora.md

The `tao_tutorials` repository is broadly classified into two components:

* **setup:** A set of quick start scripts to help you install and deploy the TAO launcher.
* **tutorials:** Beginner friendly end-to-end tutorials that will help you hit the ground running with TAO. The notebooks
  install TAO, download the required data, and run TAO commands end-to-end for various use cases.

  These tutorials are split into four categories:

   * `tao_api_starter_kit`: End-to-end notebooks that help you learn the features supported by the TAO API model of execution.
      The notebooks under the `api` directory work directly at the REST API level using REST API requests, while the `client` directory
      uses the TAO Client CLI to interact with the API server.
   * `tao_launcher_starter_kit`: Sample notebooks that walk you through the end-to-end workflow for all the
      computer-vision models supported in TAO. You can interact with TAO using the TAO launcher CLI.
   * `tao_data_services`: Sample notebooks that walk you through the end-to-end workflow of the different
      dataset manipulation and annotation tools that are included as part of TAO.
   * `tao_agent_skills_examples`: Prompt-driven tutorials that use a coding agent and the
      [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) to run TAO workflows end-to-end,
      such as data preparation, training and post-training, evaluation, AutoML tuning, and deployment.

## <a name='HowtorunTAO'></a>How to run TAO?

You can run TAO in five different ways:

* Via the [Finetuning Microservice](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_api/index.html#running-tao-finetuning-microservice)
* Via the [Launcher CLI](https://docs.nvidia.com/tao/tao-toolkit/text/quick_start_guide/running_via_launcher.html#running-tao-launcher-cli)
* Via [containers](https://docs.nvidia.com/tao/tao-toolkit/text/quick_start_guide/running_from_containers.html#running-from-containers)
* Via :ref:[Python wheels](https://docs.nvidia.com/tao/tao-toolkit/text//quick_start_guide/running_via_wheels.html#running-tao-via-wheels)
* From [source code](https://docs.nvidia.com/tao/tao-toolkit/text//quick_start_guide/advanced.html#getting-started-as-an-advanced-user)

## <a name='ImportantLinks'></a>Important Links

* [TAO Documentation](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html)

## <a name='Blogs'></a>Blogs

[Post-Train NVIDIA Cosmos 3 in One Day Using Agent Skills](https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-in-one-day-using-agent-skills/) <br>
[Train like a 'pro' with AutoML in TAO](https://developer.nvidia.com/blog/training-like-an-ai-pro-using-tao-automl/) <br>
[Deploy TAO on Azure ML](https://developer.nvidia.com/blog/creating-custom-ai-models-using-nvidia-tao-toolkit-with-azure-machine-learning/) <br>
[Synthetic Data and TAO](https://developer.nvidia.com/blog/developing-and-deploying-ai-powered-robots-with-nvidia-isaac-sim-and-nvidia-tao/) <br>
[Action Recognition Blog](https://developer.nvidia.com/blog/developing-and-deploying-your-custom-action-recognition-application-without-any-ai-expertise-using-tao-and-deepstream/) <br>
[Real-time License Plate Detection](https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/) <br>
[2 Pose Estimation: Part 1](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1/) <br>
[Part 2](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-2/) <br>
[Building ConvAI with TAO Toolkit](https://developer.nvidia.com/blog/building-and-deploying-conversational-ai-models-using-nvidia-tao-toolkit/) <br>

## <a name='License'></a> License

[TAO getting Started](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html)
License for TAO containers is included in the banner of the container. License for the pre-trained models are available with the model cards on NGC. By pulling and using the Train Adapt Optimize (TAO) Toolkit container to download models, you accept the terms and conditions of these [licenses](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/).
