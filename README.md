# On Investigating the Conservative Property of Score-Based Generative Models
[![arXiv](https://img.shields.io/badge/arXiv-2209.12753-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2209.12753)<br>

This repository contains the code implementation of the experiments presented in the paper [*On Investigating the Conservative Property of Score-Based Generative Models*](https://arxiv.org/abs/2209.12753).

<img src="assets/training.png" alt="training" width="750"/>

## Folder Organizations
- Reproduce the experimental results presented in **Section 3.1** using the code in [gaussian_example](/gaussian_example).
- Reproduce the experimental results presented in **Section 3.2** using the code in [2d_examples](/2d_examples).
- Reproduce the experimental results presented in **Section 5** using the code in [real_world](/real_world).
- Reproduce the experimental results presented in **Section 6** using the code in [autoencoder_example](/autoencoder_example).

## Dependencies
Install the necessary python packages through the following commands:

```
pip install -r requirements.txt --use-feature=2020-resolver
```

## Citing QCSBM
If you find this code useful, please consider citing our paper.
```bib
@inproceedings{chao2023investigating,
      title={On Investigating the Conservative Property of Score-Based Generative Models}, 
      author={Chen-Hao Chao and Wei-Fang Sun and Bo-Wun Cheng and Chun-Yi Lee},
      year={2023},
      booktitle={International Conference on Machine Learning (ICML)},
}
```

## License

To maintain reproducibility, we freezed the following repository and list its license below:
- [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) (at commit 1618dde) is licensed under the [Apache-2.0 License](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE)

Further changes based on the repository above are licensed under the [Apache-2.0 License](LICENSE).