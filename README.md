

# LLM

Pytorch Code of LLM method for "LLM:LEARNING-CROSS-MODALITY-PERSON-RE-IDENTIFICATION-VIA-LOW-RANK-LOCAL-MATCHING" in 
[PDF](https://ieeexplore.ieee.org/abstract/document/9521771)

### Results on the SYSU-MM01 Dataset an the RegDB Dataset 
| Method | Datasets                   | Rank@1   | Rank@10  | mAP      |
| ------ | -------------------------- | -------- | -------- | -------- |
| LLM    | #SYSU-MM01 (All-Search)    | ~ 55.25% | ~ 86.09% | ~ 52.96% |
| LLM    | #SYSU-MM01 (Indoor-Search) | ~ 59.65% | ~ 90.85% | ~ 66.46% |
| LLM    | #RegDB                     | ~ 74.85% | ~ 90.58% | ~ 71.32% |



*The code has been tested in Python 3.7, PyTorch=1.0. Both of these two datasets may have some fluctuation due to random spliting

### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py`  in to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Test.
  Test a model by
  ```bash
python test.py
  ```

  - `--dataset`: which dataset "sysu" or "regdb".

  - `--lr`: initial learning rate.
  
  - `--gpu`:  which gpu to run.

You may need manually define the data path first.

```
LLM approach model of SYSU-MMO1 in: https://pan.baidu.com/s/1ecXnB3Rhl2xQWZ2y-3HoIQ.
Code is ocjz.
```

### 3. Citation

Please kindly cite the references in your publications if it helps your research:
```
@article{jian2021llm,
  title={LLM: Learning Cross-modality Person Re-identification via Low-rank Local Matching},
  author={Jian, Feng Yu and Xu, Jing and Ji, Yi-mu and Wu, Fei},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1789-1793},
  year={2021},
  publisher={IEEE}
}
```

### 4. References

```
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.
```

```
[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
```

```
[3] M. Ye, Z. Wang, X. Lan, and P. C. Yuen. Visible thermal person reidentification via dual-constrained top-ranking. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1092–1099, 2018.
```

```
[4] Zhang H, Wu C, Zhang Z, et al. Resnest: Split-attention networks[J]. arXiv preprint arXiv:2004.08955, 2020.
```

```
[5] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021
```

