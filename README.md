# Plant Orbiter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24422932.svg)](https://doi.org/10.5281/zenodo.24422932)

Code and results for *Plant Orbiter: 3D Plant Reconstruction and Generation with Video Diffusion Orbit Cameras and 3D
Gaussian Splatting* (Hartley\*, Kok\*, Stuart, French, Pound; \* equal contribution).

One image of a plant, either a photograph or a text-to-image output, is turned into a 360° orbit video by a video
diffusion model. The orbit's cameras are known by construction, so a 3D Gaussian splat is fitted without
structure-from-motion and placed in the frame of the input photo.

![The Plant Orbiter pipeline: (1) a single photo or a text-to-image output, (2) camera poses fixed by the orbit LoRA,
(3) orbit video generation with MiniMax-H3, (4) 3D reconstruction](docs/pipeline.png)

The same steps in the code:

```
input image (photo, or Qwen-Image-2.1 from a text prompt)
 └─ pipeline/orbit.py ───────── BiRefNet cutout ─► MiniMax-H3 + 360° Orbit LoRA (ComfyUI) ─► 73-frame orbit + masks
     └─ pipeline/make_splat_kp.sh
          ├─ pipeline/known_poses.py ─ cameras known by construction: 5° per frame about the vertical axis;
          │                             elevation and focal length chosen by visual-hull consistency
          └─ nerfstudio splatfacto ─► 3D Gaussian splat
 reconstruction/  placement in the input photo's frame, scores at real held-out views, 2DGS mesh geometry, traits
```

The repository covers both experiment groups in the paper:

| Paper | What | Where |
|---|---|---|
| Sections 3-4, Tables 1-3, Figures 2-3 | Single-photo reconstruction on PlantDreamer, Botany-Bot, PLANesT-3D and GrowSplat | `pipeline/`, `reconstruction/`, `baselines/single_photo/`, `scripts/reconstruct_plants.sh` |
| Section 5, Tables 4-5, Figures 4-5 | Text-to-3D generation: fully synthetic plants and fruit counting | `pipeline/`, `baselines/text_to_3d/`, `evaluation/`, `scripts/reproduce_paper.sh`, `scripts/run_baselines.sh`, `scripts/evaluate.sh` |
| All results tables | Per-run and per-asset results, and the scripts that rebuild every table | `results/`, `tables/make_tables.py`, `evaluation/` |

## Contents

| Path | Contents |
|---|---|
| `pipeline/` | Orbit generation, known cameras, splat training, text-to-image prompts (shared by both experiment groups) |
| `reconstruction/` | Dataset conversion, input masks, placement, 2D and 3D evaluation, full-capture reference, traits, GrowSplat heights |
| `baselines/single_photo/` | Wrappers for the single-photo baselines: TRELLIS, TRELLIS.2, Hunyuan3D 2.1, SEVA, SV3D |
| `baselines/text_to_3d/` | Wrappers for the text-to-3D baselines: GaussianDreamer, DiffSplat, TRELLIS-text, MVDream, TRELLIS.2 |
| `evaluation/` | Text-to-3D evaluation: renders, appearance, realism (KD), shape (MMD-F, COV), fruit counts, counter agreement |
| `masks/` | Hybrid masks for PlantDreamer's real photos, used by both experiment groups |
| `scripts/` | Entry points: dataset download, reconstruction runs, text-to-3D runs, baselines, evaluation |
| `setup/` | Environment and model-weight setup, including every baseline at a pinned commit |
| `config/` | `paths.sh` (all locations) and the ComfyUI model-path file |
| `env/` | Pinned package lists for the environments, the one-line nerfstudio patch and the one-line Hunyuan3D patch |
| `results/` | Per-run and per-asset scores behind the paper's tables, and both counters' fruit counts (about 5 MB) |
| `tables/make_tables.py` | Rebuilds the tables from `results/` |
| `tools/` | `collect_results.py` (builds `results/` from experiment outputs), `set_input_views.py` |
| `docs/` | Figures used in this README |

## Licences

The code in this repository is released under the MIT licence (`LICENSE`). The models and datasets it uses have their
own terms, which you must accept yourself:

| Component | Licence | Notes |
|---|---|---|
| MiniMax-H3 (orbit video model) | MiniMax-H3 Community License | Does not cover the UK, EU, US or Korea by default; written approval from MiniMax is needed there. Clause V.3 forbids using its outputs to improve other AI models. Approval is personal: each user must be covered. |
| Orbit LoRA (`pablodawson/MiniMax-H3-360-Orbit-LoRA`) | Same as MiniMax-H3 | |
| Qwen-Image-2.1 | Qwen Research Licence | Non-commercial research and evaluation only. |
| BiRefNet | MIT | |
| ComfyUI, nerfstudio, gsplat | GPL-3.0, Apache-2.0, Apache-2.0 | Installed by the setup scripts, not included here. |
| Baselines: TRELLIS, TRELLIS.2, Hunyuan3D 2.1, SEVA, SV3D, GaussianDreamer, DiffSplat, MVDream | See each project and model card | Cloned by `setup/setup_<method>.sh`, not included here. Hunyuan3D 2.1 is restricted by territory. Gated on Hugging Face (accept their terms first): `stabilityai/sv3d` (SV3D), `stabilityai/stable-diffusion-3.5-medium` (DiffSplat), `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` (TRELLIS.2). |
| PlantDreamer, Botany-Bot, GrowSplat | CC BY 4.0 | The masks in `masks/` are derived from PlantDreamer and shared on its terms. |
| PLANesT-3D | CC BY-NC-ND 4.0 | Non-commercial; no redistribution of adapted material. |

## Hardware

| | Tested on | Notes |
|---|---|---|
| GPU | RTX A6000 48 GB, RTX 4090 24 GB | On 24 GB, ComfyUI streams the video model and its text encoder from RAM. Run one GPU job at a time. |
| RAM | 64 GB or more | Needed for that streaming. |
| Driver | NVIDIA with CUDA 13 support | The int8 weights need PyTorch cu130. |
| Disk | About 67 GB of weights and 25 GB of conda environments | Plus 100-200 MB per run, and the datasets. |
| OS | Linux | Not tested on Windows or macOS. |

Time per plant on an RTX A6000: orbit about 4 min, splat about 18 min with the elevation search, placement and
scoring a few minutes. Text-to-3D adds 3-4 min for the input image.

## Setup

All commands run from the repository root. Every script sources `config/paths.sh`, which puts weights, datasets and
outputs under `ORBIT_DATA` (default `./data`); export `ORBIT_DATA` first to use another disk.

```bash
source config/paths.sh
setup/setup_orbit_cam.sh                 # Python 3.12, torch cu130, diffusers, ComfyUI (commit 65787d6)
setup/setup_nerfstudio.sh                # Python 3.8, torch 2.1.2 cu118, nerfstudio (pinned + patch), gsplat 1.4.0
setup/build_gsplat.sh                    # once per GPU type, on a GPU machine
setup/setup_reconstruction_extras.sh     # piq (DISTS) and a small "masks" env with rembg
conda run --no-capture-output -n orbit_cam setup/download_weights.sh   # about 67 GB, after reading "Licences"
```

The baselines and the text-to-3D evaluation have their own environments, each built by one script at a pinned commit
(run on a GPU machine, since most compile CUDA extensions). Each script's header lists its deviations from the
project's own install steps.

| Environment | Script | Used for |
|---|---|---|
| `trellis` | `setup/setup_trellis_text.sh`, then `setup/setup_trellis_image.sh` | TRELLIS (text and image) |
| `trellis2` | `setup/setup_trellis2.sh` | TRELLIS.2 (both experiment groups) |
| `hunyuan` | `setup/setup_hunyuan.sh` | Hunyuan3D 2.1 |
| `nvs` | `setup/setup_nvs.sh` | SEVA and SV3D |
| `diffsplat`, `gaussiandreamer`, `mvdream` | `setup/setup_<method>.sh` | Text-to-3D baselines |
| `evalq`, `vqascore` | `setup/setup_evalq.sh`, `setup/setup_vqascore.sh` | ImageReward, VBench imaging, VQAScore |

Checks:

```bash
conda run -n orbit_cam python -c "import torch, diffusers; print(torch.__version__, torch.cuda.is_available())"
conda run -n nerfstudio python -c "import torch, gsplat, nerfstudio; print(torch.__version__, gsplat.__version__)"
```

If an exact pin does not resolve on your machine, relax it to the nearest available version (keep torch on the
cu130 build in `orbit_cam`) and note what changed.

| Under `$ORBIT_WEIGHTS` | File | Size | Source |
|---|---|---|---|
| `comfyui/diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21.0 GB | `Comfy-Org/MiniMax-H3` |
| `comfyui/text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB | `Comfy-Org/MiniMax-H3` |
| `comfyui/vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.8 GB | `Comfy-Org/MiniMax-H3` |
| `comfyui/vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `Comfy-Org/MiniMax-H3` |
| `comfyui/loras/` | `minimax_h3_flf2v_lora_v1.safetensors` | 0.16 GB | `pablodawson/MiniMax-H3-360-Orbit-LoRA` |
| `comfyui/background_removal/` | `birefnet.safetensors` | 0.44 GB | `Comfy-Org/BiRefNet` |
| `hf/Qwen-Image-2.1/` | full repository | 26 GB | `Qwen/Qwen-Image-2.1` (text-to-3D only) |

## Method settings

These settings are used for every result in the paper. Do not change them without noting it.

- **Orbit:** MiniMax-H3 FL2VA (int8 pruned) with the Orbit LoRA v1 at strength 1.0, 768 × 768, 73 frames, 28 steps,
  orbit seed as stated per experiment, audio off. The input image is both the first and the last keyframe, so the
  orbit closes; the last frame repeats the first and is dropped, which leaves 72 frames 5° apart. The prompt is the
  LoRA card's prompt, verbatim (`pipeline/orbit.py`). The model is guidance-distilled: no CFG, no negative prompt.
- **Cameras** (`pipeline/known_poses.py`): a level camera circling the image's vertical axis, 5° per frame, radius 1.
  Focal length and orbit elevation are chosen by silhouette consistency: a visual hull is carved from the masks and
  its recall is maximised. Elevation is searched over 0-40° in 5° steps. The hull seeds the splat.
- **Splat:** nerfstudio splatfacto (nerfstudio 1.1.5 at commit `50e0e3c` with `env/nerfstudio_splatfacto.patch`,
  gsplat 1.4.0), 30k iterations, SH degree 3, antialiased rasterisation, scale regularisation with a maximum Gaussian
  ratio of 3.
- **Single-photo reconstruction:** focal search 300-1800 px in 25 px steps; seeds 42, 1 and 2 for every plant.
- **Text-to-3D:** focal search 300-6000 px with a 3000 px fallback when the recall curve is flat (generated images
  are long-lens); text-to-image seed = asset seed, orbit seed 42.

## Single-photo reconstruction (Sections 3-4)

### Datasets

```bash
scripts/fetch_datasets.sh                 # Botany-Bot, GrowSplat, PLANesT-3D into $ORBIT_DATASETS
```

PlantDreamer (bean, kale, mint) is available from the Nottingham plant image datasets page
(https://plantimages.nottingham.ac.uk/datasets.html); unpack it to
`$ORBIT_DATASETS/PlantDreamer_Data/plants/`. bean_2 is not used: in part of its capture the masks outline a different
plant.

Conversion to a common layout (`$ORBIT_DATASETS/converted/plants/<plant>/`), run in the `nerfstudio` env:

```bash
NS="conda run --no-capture-output -n nerfstudio python"
$NS reconstruction/convert_botanybot.py
$NS reconstruction/convert_growsplat.py
$NS reconstruction/convert_planest.py
$NS reconstruction/register_planest.py                          # registers the PLANesT reference clouds to the cameras
conda run --no-capture-output -n orbit_cam python reconstruction/make_planest_masks.py
python tools/set_input_views.py                                 # the paper's input photo for each converted plant
```

PlantDreamer plants need no conversion. Their input photos and scoring views use the hybrid masks shipped in
`masks/plantdreamer_hybrid/` (the dataset mask combined with a rembg IS-Net cutout, which removes the background halo
around leaves while keeping thin stems; see `masks/README.md`).

### Run

Start ComfyUI for the orbit step, then run plants with the final settings:

```bash
conda run --no-capture-output -n orbit_cam python $COMFYUI_DIR/main.py --listen 127.0.0.1 --port 8188 \
    --extra-model-paths-config config/comfyui_extra_model_paths.yaml &

scripts/reconstruct_plants.sh pd 42 bean_3 kale_1        # dataset (pd | bb | pl), seed, plants
```

For each plant this generates the orbit from the plant's input photo, fits the splat with known cameras, places it
twice (input-view placement: from the input photo alone; oracle placement: fitted to all test views), scores both at
the real held-out views (silhouette IoU, masked PSNR, SSIM, LPIPS, DISTS; colour-corrected), and, where a reference
cloud exists, meshes the orbit with 2DGS and computes Chamfer distance and F-score after oracle placement and ICP.
Results go to `$ORBIT_OUTPUTS/<dataset>/<plant>_<input view>_s<seed>/3dgs/`.

Other parts of the evaluation:

| What | Command |
|---|---|
| Full-capture upper bound | `$NS reconstruction/fullcap_splat.py <plant> [--fold 0]` then `$NS reconstruction/fullcap_dists.py <plant> <run>` |
| Baselines | `baselines/single_photo/run_*.py` (environments in "Setup") on the inputs made by `reconstruction/make_rgba_inputs.py` (TRELLIS, TRELLIS.2, Hunyuan3D) or `reconstruction/make_nvs_inputs.py` (SEVA, SV3D; frames masked with `reconstruction/mask_frames.py`). Generators are placed with `reconstruction/align_generic.py`. |
| PLANesT traits | `reconstruction/traits_pot_masks.py` (pot masks of the input photos), then `$NS reconstruction/traits_planest.py $ORBIT_OUTPUTS/pl -o traits_planest.json --pot-cut --mesh 2dgs_v3`; summary with `reconstruction/traits_stats.py` |
| GrowSplat heights | `$NS reconstruction/growsplat_heights.py ...` (see the script's header) |

Each script documents its arguments at the top.

## Text-to-3D generation (Section 5)

```bash
scripts/reproduce_paper.sh synthetic_plants    # bean, kale, mint; seeds 1-11 and 42; 36 assets
scripts/reproduce_paper.sh fruit_counting      # tomato, lemon; N = 1-7 fruit; seeds 42, 1, 2; 42 assets
```

The prompt is the content prompt followed by a fixed framing sentence, the same for every species, seed and count:
`"A photo of {species} plants. Single potted plant, whole plant and pot fully in frame and centred with margin around
it, eye-level side view, isolated on a plain white background, studio lighting."`. For fruit counting the count is
written as a word, for example `"A photo of tomato plants with five tomatoes. ..."` (`pipeline/prompts.py`).

Each experiment writes `$ORBIT_OUTPUTS/<experiment>/images/` (the Qwen-Image inputs and `prompts.json`) and
`$ORBIT_OUTPUTS/<experiment>/splats/<image>_s42/`. To run your own prompts or images:

```bash
conda run --no-capture-output -n orbit_cam python pipeline/t2i_qwen.py --species tomato --seeds 42 1 2 --out my_run/images
scripts/images_to_splats.sh my_run/splats my_run/images/*.png
```

Any square image of one whole object on a plain background also works as input to `images_to_splats.sh`.

### Baselines

Every baseline runs at its repository's defaults, from the commit pinned in its setup script, with the same seed
numbers as ours. The prompts are in `baselines/text_to_3d/text_prompts.py`: `"A photo of {species} plants. Single
potted plant, whole plant."` (PlantDreamer's guidance prompt plus one object sentence) and, for fruit counting,
`"A photo of tomato plants with five tomatoes. Single potted plant, whole plant."`. Ours receives the same text with
the framing sentence in place of the final full stop.

| ID | Paper name | Repository @ commit | Model / configuration | Output | Experiments |
|---|---|---|---|---|---|
| `gaussiandreamer` | GaussianDreamer | `hustvl/GaussianDreamer` @ `76a38ff` | `gaussiandreamer-sd.yaml` (1200 steps, SH 0); Cap3D fine-tuned Shap-E initialisation; SD 2.1-base guidance from the `sd2-community` mirror (`check_sd21_mirror.py`) | 3DGS | both |
| `diffsplat` | DiffSplat | `chenguolin/DiffSplat` @ `a99d757` | SD3.5-medium, `gsdiff_sd35m_80g.yaml`, tag `gsdiff_gobj83k_sd35m__render`, default sampler; RaDe-GS rasteriser @ `5f5cd3b` | 3DGS | both |
| `trellis_text` | TRELLIS-text | `microsoft/TRELLIS` @ `442aa1e` | `microsoft/TRELLIS-text-xlarge`, default sampler, Gaussian decoder only; xformers attention | 3DGS | both |
| `mvdream` | MVDream | `bytedance/MVDream-threestudio` @ `efcd6c3` (+ `MVDream` @ `618f844`) | `mvdream-sd21.yaml` (10k steps, no shading), `sd-v2.1-base-4view.pt` | NeRF | both |
| `trellis2` | TRELLIS.2 | `microsoft/TRELLIS.2` @ `75fbf01` | `microsoft/TRELLIS.2-4B`, `1024_cascade`, seed 42; input = our RGBA input image (`prep_inputs.py`) | textured mesh | fruit counting (same-input control) |

```bash
scripts/run_baselines.sh synthetic_plants        # 36 assets per method
scripts/run_baselines.sh fruit_counting          # 42 per method; trellis2 needs our fruit-counting images first
```

Outputs go to `$ORBIT_OUTPUTS/<experiment>/baselines/<ID>/<asset>/`, with a `run.json` recording the prompt, seed and
time. Time per asset on an RTX A6000: TRELLIS-text 14 s, DiffSplat 27 s, GaussianDreamer 13 min, MVDream 66 min.

### Evaluation

```bash
setup/download_datasets.sh                       # PlantDreamer data (12 GB): real photos and point clouds
conda run -n orbit_cam python evaluation/prep_reference.py --out $ORBIT_DATASETS/plant_reference
scripts/evaluate.sh synthetic_plants
scripts/evaluate.sh fruit_counting               # then count the fruit by eye (below)
```

`scripts/evaluate.sh` lists each step and its environment.

- **Rendering**, identical for every method: each asset is normalised to a unit cube, made upright and rendered from
  72 views (elevations 0, 15 and 30° × 24 azimuths), 40° field of view, 512 × 512, white background. Each view is
  cropped to the subject's bounding box plus a 10% margin and padded square before scoring.
- **Synthetic plants** (n = 36 per method): CLIP ViT-L/14 score and ImageReward v1.0 against the bare prompt
  `"A photo of {species} plants"`, and VBench imaging quality, each averaged over the 72 views. Realism: kernel
  distance (KD) to PlantDreamer's real photos (cut out with `masks/plantdreamer_hybrid/`, camera elevations −5° to
  35°), on Inception-v3 and DINOv2-base features, as the mean over species with a 95% bootstrap interval
  (`kd_bootstrap.py`). Shape: depth points at the same 72 cameras against the real point clouds, both scaled to unit
  height and aligned by a yaw sweep and rigid ICP; F-score at τ = 2% of height, MMD-F and COV (`shape_metrics.py`).
  Tests: unpaired Mann-Whitney U against ours (`summarise_metrics.py`); KD uses the bootstrap p.
- **Fruit counting** (n = 42 per method): fruit are counted by eye, blind to the requested N, from a contact sheet of
  36 views and a turntable video per asset (`count_sheets.py`), following the rules in the paper. Exact (%), MAE and,
  for ours and TRELLIS.2, the share whose 3D count equals the input image's count. CLIP, ImageReward and VQAScore use
  the prompt without the count. Tests: Fisher's exact test and Mann-Whitney U, unpaired against ours. A second
  counter, blind to method, asset and N, recounted 84 of the 252 assets and 14 of the 42 input images
  (`count_agreement.py`: exact agreement, mean absolute difference, Krippendorff's α).

### Output per run

| Path | Contents |
|---|---|
| `orbit.mp4`, `cutout.png` | Orbit video and the background-removed input |
| `frames/`, `masks/`, `rgba/` | 73 orbit frames, BiRefNet masks (255 = subject), RGBA frames |
| `3dgs/known_poses.json` | Chosen focal length and elevation, hull recall, search curves |
| `3dgs/sparse/0/` | Cameras and hull seed points in COLMAP format |
| `3dgs/export/splat.ply` | The 3D Gaussian splat |
| `3dgs/score_{input,oracle}_cc.json`, `dists_*_cc.json` | Reconstruction only: scores at the real held-out views |
| `3dgs/2dgs_v3/mesh.ply`, `gao_oracle_v3.json` | Reconstruction only: 2DGS mesh and 3D geometry scores |

## Results and tables

`results/` holds the per-run results behind every table, collected from the experiment outputs by
`tools/collect_results.py`:

| File | Contents |
|---|---|
| `scores_2d.csv` | Per dataset, plant, method, seed and placement: silhouette IoU, masked PSNR, SSIM, LPIPS, DISTS |
| `scores_2d_per_view.csv.gz` | The same, per held-out view |
| `geometry_3d.csv` | Chamfer distance (×10³) and F-score per plant, method and seed |
| `placement_generators.csv` | Rotation between input-view and oracle placement of the camera-less generators |
| `orbit_cameras.csv` | Focal length and elevation chosen for each of our runs |
| `full_capture.csv` | Full-capture upper bound per plant and fold |
| `ablations.csv` | Elevation search against a level orbit; estimated against known focal length |
| `growsplat_iou.csv` | GrowSplat silhouette IoU per date and method |
| `traits_planest_potcut2_v3.json`, `heights_input_zoom_v3.json` | Trait and height measurements against the reference clouds |
| `input_views.csv` | The input photo used for each plant |

Text-to-3D results (method IDs as in "Baselines", plus `ours`, `real` for the real-to-real shape reference and
`input` for our input images):

| File | Contents |
|---|---|
| `synthetic_plants/per_asset.csv`, `quality.csv` | Per method and asset: CLIP, ImageReward, VBench imaging (and unreported extras) |
| `synthetic_plants/realism.csv` | KD per method, species and feature, with bootstrap interval and p against ours |
| `synthetic_plants/shape_pairs.csv`, `shape.csv` | F-scores per asset and real plant; MMD-F and COV per method |
| `fruit_counting/per_asset.csv`, `quality.csv`, `vqascore.csv` | Appearance scores per method and asset |
| `fruit_counting/input_image/` | The same scores on the input images |
| `fruit_counting/counts_counter1.csv` | First counter's count for every asset, and the input image's count on our rows |
| `fruit_counting/counts_counter2.csv` | Second counter's recount of the subset |

```bash
python tables/make_tables.py            # Tables 1-3 and the reconstruction results (needs numpy and scipy)

# Tables 4 and 5
python evaluation/summarise_metrics.py results/synthetic_plants        # appearance means and tests
python evaluation/count_metrics.py results/fruit_counting/counts_counter1.csv \
    --quality results/fruit_counting results/fruit_counting/input_image  # fruit-count table
python evaluation/count_agreement.py results/fruit_counting/counts_counter2.csv \
    --counts results/fruit_counting/counts_counter1.csv                 # agreement: 91.7%, alpha = 0.984
```

These write `summary.md`, `count_metrics.md`, `count_table.tex` and `agreement.md` next to their inputs. KD, MMD-F
and COV are read directly from `realism.csv` and `shape.csv` (species `all`; Table 4 shows KD-Inception × 100).

## Practical notes

- Inputs should be square, centred, show the whole subject on a plain background, and be taken at roughly eye level.
  The orbit circles the image's vertical axis; the elevation search absorbs inputs taken from up to 40° above.
- Background removal is required. Without it the background stays fixed while the subject turns.
- Thin structures (grasses, cereals) are the hard case. The Orbit LoRA was trained on human subjects, so plants are out
  of its training distribution; occasional wobble or growing leaves can appear.
- ComfyUI keeps about 15 GB of VRAM after a run. `images_to_splats.sh` stops its own server before training; if you
  run ComfyUI yourself, free it first:
  `curl -X POST 127.0.0.1:8188/free -H "Content-Type: application/json" -d '{"unload_models":true,"free_memory":true}'`.
- `crypt.h: No such file or directory` when compiling gsplat: `setup_nerfstudio.sh` installs `libxcrypt` and sets
  `CUDA_HOME` and `CPATH`; set them the same way if you build the environment differently.
- A missing ComfyUI node (`MiniMaxH3ImageToVideo`, `RemoveBackground`) means the ComfyUI build is too old; check out
  commit `65787d6`.
- `conda run` does not pass a heredoc on stdin; write the script to a file instead.
- Do not edit a shell script while a job is running it: bash reads scripts as it executes them.

## Citation

Hartley, Z. K. J.\*, Kok, Y. E.\*, Stuart, L. A. G., French, A. P., Pound, M. P. *Plant Orbiter: 3D Plant
Reconstruction and Generation with Video Diffusion Orbit Cameras and 3D Gaussian Splatting.* (\* equal contribution)
Submitted to *Plant Phenomics*. arXiv: [2611.12739](https://arxiv.org/abs/2611.12739).

Code and data: Zenodo, [doi:10.5281/zenodo.24422932](https://doi.org/10.5281/zenodo.24422932) (all versions).

```bibtex
@article{hartley2026plantorbiter,
  title   = {Plant Orbiter: 3D Plant Reconstruction and Generation with Video Diffusion Orbit Cameras and
             3D Gaussian Splatting},
  author  = {Hartley, Zane K. J. and Kok, Yong En and Stuart, Lewis A. G. and French, Andrew P. and Pound, Michael P.},
  journal = {arXiv preprint arXiv:2611.12739},
  year    = {2026}
}
```
