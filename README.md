
# <h1 align="center">SldprtNet: A Large-Scale Multimodal Datasetfor CAD Generation in Language-Driven 3D Design
<div align="center">
 <img width="800" height="350" alt="Cover"         
   src="https://github.com/user-attachments/assets/a7798122-b213-44eb-ba3d-4ad572e96e5d"
   />
   </div>

## Overview

We introduce **SldprtNet**, a large-scale dataset comprising over **242,000 industrial parts**, designed for:

- Semantic-driven CAD modeling  
- Geometric deep learning  
- Training and fine-tuning multimodal models for 3D design

The dataset provides 3D models in both `.step` and `.sldprt` formats. To enable parametric modeling and scalable transformation, we developed two tools: **encoder** and **decoder** , which supporting 13 types of CAD commands.

Each sample also includes:
- A composite rendered image from 7 viewpoints (`image`)  
- A parameterized modeling script (`Encoder_Text`)  
- A natural language description (`Des_Text`)

In this project, we use **Qwen2.5-VL-7B** to generate the descriptions and manually verify alignment across all modalities. The result is a fully aligned, comprehensive multimodal dataset for semantic-driven CAD and cross-modal learning.



## Directory structure
```
📁 SLDPRT-NET
|--- .gitattributes
|--- README.md
|--- batch_infer.py				 				# Inference from Qwen2.5-7B-VL to generate descriptions
|--- encoder.swp				 				# Macro1 for parameterizing CAD models
|--- decoder.swp				 				# Macro2 for reconstruct CAD models
|--- sub_dataset.rar
|   |--- Des_Text                				# Generated natural language descriptions
|   |--- Encoder_Text            				# CAD commands sequences
|   |--- image                   				# Rendered part images (multi-view)
|   |--- model_sldprt            				# Original SolidWorks models (.sldprt)
|   |--- model_step              				# Converted STEP models (.step)
|
|--- demo                        				# Demo samples from dataset
|   |--- des_text
|   |--- encoder_text
|   |--- image
|   |--- part
```

## Environment
### Inference
- OS: Linux  
- Python: 3.10  
- CUDA: 12.2  
- GPU: NVIDIA A100 80GB ×4 *(Recommended configuration)*  
- Model: Qwen2.5-7B-VL
- PyTorch: `torch==2.5.1+cu121`  
- Transformers: `transformers==4.51.0`  
- Accelerate: `accelerate==1.8.1`  
- Pillow: `Pillow==11.2.1`  
- NumPy: `numpy==1.24.4`

``` python
# Inference launch command
# --start-id and --end-id define the range of file IDs to process

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="bf16" batch_infer.py --start-id 000000 --end-id 242606
```
### Baseline

## Dataset
<img width="4135" height="615" alt="Datasetshow" src="https://github.com/user-attachments/assets/1d81ca23-5d14-4712-bd30-e2c95ccf917d" />
The dataset is currently under review. We provide a subset of the SldprtNet dataset, named `sub_dataset`, which contains 1,000 samples for public access and testing purposes.

This subset includes the following components:
- CAD models in both `.sldprt` and `.step` formats
- Rendered images from multiple views(`image`)
- Parameterized modeling scripts (`Encoder_Text`)
- Natural language descriptions (`Des_Text`)


## Tools
We developed two tools, **encoder** and **decoder**, which together support the parameterization and reconstruction of the 13 types of CAD commands listed in the table below.
| **Chamfer**       | **Sketch2D**       | **RefAxis**        | **Extrusion**      | **Revolution**       | **Linear Pattern**   | **HoleWzd**       |
|-------------------|-------------------|-------------------|-------------------|---------------------|---------------------|-----------------------|
| **Fillet**  | **Sketch3D**       | **RefPlane**       | **Cut-Extrusion**      | **Cut-Revolution**   | **Mirror Pattern** |                       |

**Encoder**: Losslessly converts `.sldprt` files into `.txt` files containing CAD commands and parameters, as shown in `Macro1`.
**Decoder**: Reconstructs `.sldprt` files from the parameterized `encoder_text` in `.txt` format, as shown in `Macro2`.

Click here to watch a video demonstrating the correct usage of the tools, as well as the parameterization and reconstruction process using the demo.
