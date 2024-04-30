# R2L: Distilling NeRF to NeLF

### [Project](https://snap-research.github.io/R2L/) | [ArXiv](https://arxiv.org/abs/2203.17261) | [PDF](https://arxiv.org/pdf/2203.17261.pdf) 

<div align="center">
    <a><img src="figs/snap.svg"  height="120px" ></a>
    &nbsp
    <a><img src="figs/smile.png"  height="100px" ></a>
</div>

This repository is for the new neral light field (NeLF) method introduced in the following ECCV'22 paper:
> **[R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis](https://snap-research.github.io/R2L/)** \
> [Huan Wang](http://huanwang.tech/) <sup>1,2</sup>, [Jian Ren](https://alanspike.github.io/) <sup>1</sup>, [Zeng Huang](https://zeng.science/) <sup>1</sup>, [Kyle Olszewski](https://kyleolsz.github.io/) <sup>1</sup>, [Menglei Chai](https://mlchai.com/) <sup>1</sup>, [Yun Fu](http://www1.ece.neu.edu/~yunfu/) <sup>2</sup>, and [Sergey Tulyakov](http://www.stulyakov.com/) <sup>1</sup> \
> <sup>1</sup> Snap Inc. <sup>2</sup> Northeastern University \
> Work done when Huan was an intern at Snap Inc.

**[TL;DR]** We present R2L, a deep (88-layer) residual MLP network that can represent the neural *light* field (NeLF) of complex synthetic and real-world scenes. It is featured by compact representation size (~20MB storage size), faster rendering speed (~30x speedup than NeRF), significantly improved visual quality (1.4dB boost than NeRF), with no whistles and bells (no special data structure or parallelism required).

<div align="center">
    <a><img src="figs/frontpage.png"  width="700" ></a>
</div>


## Reproducing Our Results
Below we only show the example of scene `lego`. There are other scenes in the original repo, but for simplicity, I just use lego.

### 0. Download the code
```
git clone https://github.com/AA137/R2L-quant.git && cd R2L
```


### 1. Set up (original) data
```bash
sh scripts/download_example_data.sh
```

### 2. Set up environment with Anaconda
- `salloc --gres=gpu --mem=32GB`
- `conda create --name R2L python=3.9.6`
- `conda activate R2L`
- `pip install -r requirements.txt` (We use torch 1.9.0, torchvision 0.10.0)
- `pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3`

### 3. Quick start: test trained models

- Run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --trial.ON --trial.body_arch resmlp --pretrained_ckpt lego.tar --render_only --render_test --testskip 1 --experiment_name Test__R2L_W256D88__blender_lego
```
This should give the output with 32-bit precision.
Alternatively to test this process on other models, you can do 
- Download models:
```
sh scripts/download_R2L_models.sh
```
to get models for other scenes as well. 

### 4. Test: 16-bit precision

`python precision_change_16.py`
Then, run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --trial.ON --trial.body_arch resmlp --pretrained_ckpt lego_quant16.tar --render_only --render_test --testskip 1 --experiment_name Test__R2L_W256D88__blender_lego
```

### 4. Test: 8-bit precision (fake)

`python precision_changer.py`
Then, run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --trial.ON --trial.body_arch resmlp --pretrained_ckpt lego_quant8.tar --render_only --render_test --testskip 1 --experiment_name Test__R2L_W256D88__blender_lego
```
### 5. Test: 8-bit precision (real)

In main.py, add `quant_modules.initialize()` as the first line after 
`if __name__ == '__main__':`
Then, run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --trial.ON --trial.body_arch resmlp --pretrained_ckpt lego_quant8.tar --render_only --render_test --testskip 1 --experiment_name Test__R2L_W256D88__blender_lego
