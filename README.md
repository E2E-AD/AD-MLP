# Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes
This repository is an official implementation of the technical report [AD-MLP](https://arxiv.org/pdf/2305.10430.pdf) 

<br/>

> Jiang-Tian Zhai\*, Feng Ze\*, Jinhao Du\*, Yongqiang Mao\*, Jiang-Jiang Liu&#8224;, Zichang Tan, Yifu Zhang, Xiaoqing Ye, Jingdong Wang&#8224;
> 
> Baidu Inc.
>
> \*: equal contribution, <sup>&#8224;</sup>: corresponding author.
>

## News
* 2023.05.18: Paper is released!
* 2023.05.17: Code / Models are released!

## Introduction

<div align="center">
<img src="./pipeline.png" />
</div>


- We design an MLP-based method that takes raw sensor data as input and directly outputs the future trajectory of the ego vehicle, without using any perception or prediction information such as camera images or LiDAR. 
- This simple method achieves state-of-the-art end-to-end planning performance on the nuScenes dataset, reducing the average L2 error by about 30\%.
- We hope our findings are helpful to other researchers in this area.

## Results
- Open-loop planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit). 

| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: |
| ST-P3 | 1.33 | 2.11 | 2.90 | 0.23 | 0.62 | 1.27 |
| UniAD | 0.48 | 0.96 | 1.65 | **0.05** | 0.17 | 0.71 |
| VAD-Tiny | 0.20 | 0.38 | 0.65 | 0.10 | 0.12 | 0.27 |
| VAD-Base | 0.17 | 0.34 | 0.60 | 0.07 | **0.10** | 0.24 |
| Ours | **0.14** | **0.10** | **0.41** | 0.10 | **0.10** | **0.17** |

## Get Started

* Environment
  Linux, Python==3.7.9, CUDA == 11.2, pytorch == 1.9.1 or paddlepaddle == 2.3.2. Besides, follow instruction in ST-P3 for running its evaluation process.
  ```
  cd deps/stp3
  conda env create -f environment.yml
  ```

* Prepare Data   
Download the [nuScenes](https://www.nuscenes.org/download) Dataset.

* Pretrained weights   
To verify the performance on the nuScenes Dataset, we provide the pretrained model weights ([Google Drive](https://drive.google.com/drive/folders/1CJa54-Ft8qakR4EyRtxvswQxT1dgPB_9) and [Baidu Netdisk](https://pan.baidu.com/s/1cEDETxG-HHwyC7ATBk_hyQ?pwd=9fbf)). Please download them (paddle checkpoint, token of validation set...) to the root directory of this project.

* Paddle Evaluation   
  ```
  python paddle/model/AD-MLP.py
  python deps/stp3/evaluate_for_mlp.py
  ```
  The first line saves the predicted 6 frames' trajectories of the next 3s in output_data.pkl. And the second line applies the ST-P3 evaluation on it. The final evaluation output contains   the L2 error and collision rate in the next 1,2 and 3s.


## Contact
If you have any questions or suggestions about this repo, please feel free to contact us (jtzhai30@gmail.com, j04.liu@gmail.com, wangjingdong@outlook.com).

## Acknowledgement
This repo is build based on [ST-P3](https://github.com/OpenPerceptionX/ST-P3). Thanks for their great work.

## License
All code in this repository is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### BibTeX

If you find our work and this repository useful. Please consider giving a star and citation.

```bibtex
@article{zhai2023ADMLP,
  title={Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes},
  author={Zhai, Jiang-Tian and Feng, Ze and Du, Jihao and Mao, Yongqiang and Liu, Jiang-Jiang and Tan, Zichang and Yifu Zhang and Ye, Xiaoqing and Wang, Jingdong},
  journal={arXiv preprint arXiv:2305.10430},
  year={2023}
}
```
