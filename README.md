## A General Descent Aggregation Framework for Gradient-based Bi-level Optimization
This repo contains code accompaning the paper, [A General Descent Aggregation Framework for Gradient-based Bi-level Optimization (Liu et al., ICML 2020)](https://arxiv.org/abs/2006.04045). It includes code for running the few-shot classification and data hyper-cleaning experiments.

### Abstract
In recent years, a variety of gradient-based first-order methods have been developed to solve bi-level optimization problems for learning applications. However, theoretical guarantees of these existing approaches heavily rely on the simplification that for each fixed upper-level variable, the lower-level solution must be a singleton (a.k.a., Lower-Level Singleton, LLS). In this work, we first design a counter-example to illustrate the invalidation of such LLS condition. Then by formulating BLPs from the view point of optimistic bi-level and aggregating hierarchical objective information, we establish Bi-level Descent Aggregation (BDA), a flexible and modularized algorithmic framework for generic bi-level optimization. Theoretically, we derive a new methodology to prove the convergence of BDA without the LLS condition. Our investigations also demonstrate that BDA is indeed compatible to a verify of particular first-order computation modules. Additionally, as an interesting byproduct, we also improve these conventional first-order bi-level schemes (under the LLS simplification). Particularly, we establish their convergences with weaker assumptions. Extensive experiments justify our theoretical results and demonstrate the superiority of the proposed BDA for different tasks, including hyper-parameter optimization and meta learning.

### Dependencies
This repository is mainly developed based on the [BOML](BOML - A Bilevel Optimization Library in Python for Meta Learning) Code base.
You can simply run the following command automatically install the dependencies

```pip install -r requirement.txt ```


###  Data Preparation

You can download the [omniglot](https://github.com/brendenlake/omniglot), 
[miniimagenet](https://github.com/renmengye/few-shot-ssl-public/), [mnist](http://yann.lecun.com/exdb/mnist/) and [fashionmnist](https://github.com/zalandoresearch/fashion-mnist) dataset from the attached link, and put the dataset in the corresponding root folder.

### Usage

You can run the python file for different applications following the script below:

```
cd test_script
Python Few_shot.py --classes=5 --examples_train=1 --examples_test=1 --meta_batch_size=1 --alpha=0.4 # For few shot classification tasks.
Python  Data_hyper_cleaning.py # For data hyper-cleaning tasks.
```
As for the few shot classification tasks, the default parameters are for 5 way 1 shot classification on omniglot. You can modify the `script_helper.py` for more different settings. 
 
### Citation

If you use BDA for academic research, you are highly encouraged to cite the following paper:
- Risheng Liu, Pan Mu, Xiaoming Yuan, Shangzhi Zeng, Jin Zhang. ["A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton"](https://arxiv.org/abs/2006.04045). ICML, 2020.
- Yaohua Liu, Riseng Liu. ["Towards Gradient-based Bilevel Optimization with Non-convex Followers and Beyond"](https://arxiv.org/abs/2110.00455). ICMEW, 2021.

### License 

MIT License

Copyright (c) 2021 Vision Optimizaion Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
