# Is Perception Information Necessary in End-to-End Autonomous Driving? An Empirical Study on the nuScenes Dataset


<br/>

> Jiang-Tian Zhai\*, Feng Ze\*, Jinhao Du\*, Yongqiang Mao\*, Jiang-Jiang Liu&#8224;, Zichang Tan, Yifu Zhang, Xiaoqing Ye, Jingdong Wang&#8224;
> 
> Baidu Inc.
>
> \*: equal contribution, <sup>&#8224;</sup>: corresponding author.
>

## News
* Code/Models are coming soon. Please stay tuned!

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


## Contact
If you have any questions or suggestions about this repo, please feel free to contact us (jtzhai30@gmail.com, j04.liu@gmail.com, wangjingdong@outlook.com).

## References
This repo is build based on [ST-P3](https://github.com/OpenPerceptionX/ST-P3). Thanks for their great work.

## License
All code in this repository is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### BibTeX

If you find our work and this repository useful. Please consider giving a star and citation.

```bibtex
@article{zhai2023,
  title={},
  author={Zhai, Jiang-Tian and Feng, Ze and Du, Jihao and Mao, Yongqiang and Liu, Jiang-Jiang and Tan, Zichang and Ye, Xiaoqing and Wang, Jingdong},
  journal={Arxiv},
}
```