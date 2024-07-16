# YOLOv8 无人机追踪

在在自制数据集上上进行微调，实现无人机追踪，并且通过channel剪枝压缩模型并部署



## TRAIN.py - 用于BN层的稀疏化训练

> 在训练过程中通过在trainer中添加
>
> ```python
> l1_lambda = 1e-2 * (1 - 0.9 * epoch / self.epochs)
> for k, m in self.model.named_modules():
>     if isinstance(m, nn.BatchNorm2d):
>         m.weight.grad.data.add_(l1_lambda * torch.sign(m.weight.data))
>         m.bias.grad.data.add_(1e-2 * torch.sign(m.bias.data))
> ```
>
> 使用L1正则化对YOLO模型的BatchNorm层的权重和偏置进行约束，通过惩罚模型权重的绝对值来鼓励稀疏性。



## PRUNE.py - 用于针对channel的剪枝

> 通过对BatchNorm层的权重进行排序并设置阈值，剪去那些权重较小的卷积核。将所有BatchNorm层的权重绝对值拼接成一个大的张量，然后按照从大到小的顺序进行排序。根据预设的剪枝因子（如0.85），确定一个阈值，即排序后第85%的位置上的权重值。这个阈值用于决定哪些权重会被剪除。



## TRAIN.py - 剪枝后的重训练

> 移除原来在BN层添加的正则化



## TEST.py vs ONNX.py - 分别加载pt和onnx文件测试结果