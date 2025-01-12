# 项目说明

深度FBM有两种训练思路，其中一种是逐层训练：

> from train.train import DFBM

另一种是整体推理，逐层计算损失，进行网络的训练：

> from train.train import pre_DFBM

这个项目用以说明，深层的FBM能够有效的提升准确率。

# 文件

* 以"old"开头的文件，表示训练集选取过大，并且使用逐层训练的模式。


