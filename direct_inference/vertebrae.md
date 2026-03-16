<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Apply to Vertebrae Segmentation</h3>
<p align="center">
    <a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
    <a href='https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> 
    <a href='document/promotion_slides.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a> 
    <a href='document/dom_wse_poster.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a> 
    <a href='https://www.cs.jhu.edu/news/ai-and-radiologists-unite-to-map-the-abdomen/'><img src='https://img.shields.io/badge/WSE-News-yellow'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/SuPreM"><img src="https://img.shields.io/github/stars/MrGiovanni/SuPreM?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

##### 0. Download CT scans

```bash
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
```

The CT scans are organized in such a way:

```
AbdomenAtlasDemo
    ├── BDMAP_00000006
    │   └── ct.nii.gz
    ├── BDMAP_00000031
    │   └── ct.nii.gz
```

##### 1. Clone and setup the GitHub repository
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
cd ..
```

<details>
<summary style="margin-left: 25px;">[Option] if you get certificate issues when using wget</summary>
<div style="margin-left: 25px;">

```bash
wget --no-check-certificate http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
```

</div>
</details>


##### 2 Create environments

**Option A – Original stack (Python 3.9, PyTorch 1.11, CUDA 11.3)**  
Use this for older GPUs or if you prefer the exact versions from the paper.
```bash
conda create -n suprem python=3.9
source activate suprem
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

**Option B – RTX 50 series / Blackwell (e.g. RTX 5060 Ti)**  
Use this if you have an NVIDIA RTX 50-series GPU. PyTorch 1.11 does not include kernels for Blackwell, so you need PyTorch 2.7+ with CUDA 12.8.
```bash
conda create -n suprem50 python=3.10
conda activate suprem50
cd SuPreM/direct_inference/

# PyTorch 2.7 with CUDA 12.8 (Blackwell support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# MONAI compatible with PyTorch 2.x (same model APIs used here)
pip install "monai[all]>=1.4"

# Rest of dependencies (no torch/monai in this file)
pip install -r requirements-gpu-rtx50.txt
```
Then run inference **without** `--cpu`; the same checkpoint and commands work. If you see any MONAI import or API errors, report them and we can adjust the code.

<details>
<summary style="margin-left: 25px;">[Troubleshooting] pip fails on pytorch-lightning (invalid metadata / ".* suffix")</summary>
<div style="margin-left: 25px;">

If you see an error like `pytorch-lightning==1.6.4 has invalid metadata: .* suffix can only be used with == or !=`, your pip is 24.1 or newer, which rejects that metadata. Use either:

**Option A – Use an older pip (recommended if you need the exact 1.6.x line):**
```bash
pip install "pip<24.1"
pip install -r requirements.txt
```

**Option B – Keep current pip:**  
`requirements.txt` pins `pytorch-lightning>=1.6.4,<1.7` so pip can install 1.6.5, which is compatible and often has valid metadata.

</div>
</details>

<details>
<summary style="margin-left: 25px;">[Troubleshooting] RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED</summary>
<div style="margin-left: 25px;">

This can happen when the system CUDA/cuDNN driver doesn’t match PyTorch 1.11’s bundled cuDNN. Try either:

**Option A – Disable cuDNN (still use GPU):**
```bash
SUPREM_DISABLE_CUDNN=1 python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize
```

**Option B – Run on CPU only (slower):**
```bash
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize --cpu
```

</div>
</details>

<details>
<summary style="margin-left: 25px;">[Troubleshooting] CUDA error: no kernel image is available for execution on the device</summary>
<div style="margin-left: 25px;">

PyTorch 1.11.0+cu113 was built for older GPU architectures. If your GPU is newer (e.g. **RTX 5060 Ti**, RTX 40xx, or Blackwell), there are no compiled kernels for it in that build.

- **RTX 50 series (e.g. RTX 5060 Ti):** Use **Option B** in step 2 above: a separate env with PyTorch 2.7+ and CUDA 12.8 (`requirements-gpu-rtx50.txt`). Then run inference without `--cpu`.
- **Otherwise:** Use CPU for now:  
  `python -W ignore inference.py ... --customize --cpu`

</div>
</details>

##### 3. Generate vertebrae masks by the AI

```bash
datarootpath=/path/to/your/AbdomenAtlasDemo # NEED MODIFICATION!!!

pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict

python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize
```

The vertebrae masks will be saved as
```
AbdomenAtlasDemoPredict
    ├── BDMAP_00000006
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
    ├── BDMAP_00000031
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
```

##### 4. [Important!] Postprocess vertebrae masks

Check the AI-predicted vertebrae masks (`combined_labels.nii.gz`) and the original CT scans (`ct.nii.gz`) using software such as [ITK-SNAP](https://www.itksnap.org/pmwiki/pmwiki.php). If you look closely at the AI-predicted masks, you will see many errors. Please design an automatic postprocessing to reduce these errors as many as you can. The postprocessing should be formatted in a separated python file `postprocessing_vertebrae.py`.

This is an illustration of vertebrae (and rib) label refinement.

![Refinement](https://github.com/MrGiovanni/SuPreM/blob/main/document/LetsSegmentVertebrae.png)
</div>

To identify the errors, you will need some knowledge about vertebrae in the human body as follow.

![Vertebral anatomy](https://i0.wp.com/aneskey.com/wp-content/uploads/2023/08/f01-01-9780323882262.jpg)
</div>

```bash
# Basic postprocessing
python postprocessing_vertebrae.py \
    --input_dir ./AbdomenAtlasDemoPredict \
    --output_dir ./AbdomenAtlasDemoPredict_refined

# With CT-guided bone-mask refinement (recommended for best results)
python postprocessing_vertebrae.py \
    --input_dir ./AbdomenAtlasDemoPredict \
    --output_dir ./AbdomenAtlasDemoPredict_refined \
    --ct_root_path /path/to/AbdomenAtlasDemo
```

The postprocessing pipeline follows a conservative "first, do no harm" philosophy inspired by ShapeKit [1] and anatomic consistency priors from Meng et al. [3]. It applies (in order):

1. **Per-label connected-component cleanup** — keeps only the largest connected component per vertebra, removes tiny fragments (< 100 voxels by default).
2. **Adjacent-triplet fragment reassignment** (from ShapeKit [1]) — for each vertebra, checks if any of its connected components are spatially closer to a neighboring vertebra's main body and reassigns them. Fixes fragments assigned to the wrong adjacent vertebra without global operations.
3. **Overlap resolution via distance transforms** — where multiple labels claim the same voxel, the voxel is assigned to the label it is deepest inside (Euclidean distance to boundary), producing smooth Voronoi-like inter-vertebra boundaries.
4. **Cautious label reordering** — only swaps adjacent label pairs whose SI centroids are clearly reversed (by more than half the median inter-vertebra gap). Never cascades: at most one swap per pair per pass.
5. **Statistical validation** (from Meng et al. [3]) — checks vertebra volumes against learned regressors conditioned on spine level (cervical/thoracic/lumbar) and inter-vertebral distances against Gaussian priors. Warns about violations and optionally splits oversized vertebrae at the SI midpoint.
6. **Hole filling** — fills internal holes within each vertebra mask using `binary_fill_holes`. No morphological closing or opening (which can destroy small cervical vertebrae).
7. **Gentle Gaussian label smoothing** — each label's binary mask is Gaussian-blurred (σ=0.5) and the highest-response label wins at each voxel, with a minimum confidence threshold of 0.2 to prevent label bleeding.
8. **(Optional) CT-guided bone-mask refinement** — when `--ct_root_path` is provided, vertebra masks are intersected with the CT bone-intensity region (HU ∈ [150, 3000] by default, with dilation for partial-volume tolerance), removing soft-tissue false positives.
9. **Sanity checks** — warns about anomalous volumes and large spatial gaps between consecutive vertebrae.

Three legacy steps are available but **disabled by default** (they caused severe DSC regression in testing): spine-centerline outlier removal (`--enable_outlier_removal`), spine-envelope gap filling (`--enable_gap_fill`), and missing-vertebra interpolation (`--enable_interpolation`).

The refined masks are saved in the same directory structure as the original predictions.

##### Related Work

[1] Liu, Junqi, Dongli He, Wenxuan Li, Ningyu Wang, Alan L. Yuille, and Zongwei Zhou. ["ShapeKit."](https://www.cs.jhu.edu/~zongwei/publication/liu2025shapekit.pdf) In International Workshop on Shape in Medical Imaging, pp. 44-58. Cham: Springer Nature Switzerland, 2025. 

[3] Meng, Di, Edmond Boyer, and Sergi Pujades. ["Vertebrae localization, segmentation and identification using a graph optimization and an anatomic consistency cycle."](https://gitlab.inria.fr/spine/vertebrae_segmentation) Computerized Medical Imaging and Graphics 107 (2023): 102235.

[2] Jaus, Alexander, Constantin Seibold, Kelsey Hermann, Negar Shahamiri, Alexandra Walter, Kristina Giske, Johannes Haubold, Jens Kleesiek, and Rainer Stiefelhagen. ["Towards unifying anatomy segmentation: Automated generation of a full-body ct dataset."](https://github.com/alexanderjaus/AtlasDataset) In 2024 IEEE International Conference on Image Processing (ICIP), pp. 41-47. IEEE, 2024.
