# WDSR

This repository is implementation of the ["Wide Activation for Efficient and Accurate Image Super-Resolution"](https://arxiv.org/abs/1808.08718).

参考：https://github.com/yjn870/WDSR-pytorch

## Requirements

- paddlepaddle 2.4.0
- Numpy 1.15.4
- Pillow-SIMD 5.3.0.post1
- h5py 2.8.0
- tqdm 4.30.0

## Prepare dataset

To prepare dataset used in experiments, first download dataset files from this [link](https://data.vision.ee.ethz.ch/cvl/DIV2K) and organize it as shown below.

```bash
/YOUR_STORAGE_PATH/DIV2K
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
├── DIV2K_valid_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_train_HR.zip
├── DIV2K_train_LR_bicubic_X2.zip
├── DIV2K_train_LR_bicubic_X3.zip
├── DIV2K_train_LR_bicubic_X4.zip
├── DIV2K_valid_HR.zip
├── DIV2K_valid_LR_bicubic_X2.zip
├── DIV2K_valid_LR_bicubic_X3.zip
└── DIV2K_valid_LR_bicubic_X4.zip
```

By default, we use "0001-0800.png" images to train the model and "0801-0900.png" images to validate the training.
All experiments also use images with BICUBIC degradation on RGB space.

## Training

### WDSR Baseline (=WDSR-A) Example

```bash
python train.py --dataset-dir "/root/autodl-tmp/paddle_SR/SR/DATA/DIV2K" \
                --output-dir "/root/autodl-tmp/paddle_SR/SR/WDSR/outputs_a" \
                --model "WDSR-A" \
                --scale 2 \
                --n-feats 32 \
                --n-res-blocks 16 \
                --expansion-ratio 4 \
                --res-scale 1.0 \
                --lr 1e-3
```
If you want to modify more options, see the `core/option.py` file.



## Evaluation

Trained model is evaluated on DIV2K validation 100 images.

```bash
WDSR-A
python eval.py --model "WDSR-A" \
               --dataset-dir "/root/autodl-tmp/paddle_SR/SR/DATA/DIV2K" \
               --checkpoint-file "/root/autodl-tmp/paddle_SR/SR/WDSR/outputs_a/WDSR-A-f32-b16-r4-x2-best.pdiparams.tar"
               
```

| Model | Scale | Residual Blocks | Parameters | PSNR |
|-------|-------|-----------------|------------|------|
| WDSR  | x2    | 16              | 1.19M      | 34.36 dB |
