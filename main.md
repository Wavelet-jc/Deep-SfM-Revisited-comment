personal access token: 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
## 权重衰减
权重衰减（Weight Decay）是一种常用的正则化技术，用于在训练神经网络时减少过拟合现象。它通过在损失函数中添加一个正则化项来约束模型的权重参数。
具体而言，权重衰减通过在损失函数中添加一个L2正则化项来实现。L2正则化项通常被定义为权重参数的平方和与一个正则化因子的乘积。正则化因子控制了正则化的强度。
添加了权重衰减的损失函数可以写作：
loss_with_weight_decay = loss + λ * regularization_term
其中，loss是原始的损失函数，λ是正则化因子，regularization_term是权重参数的平方和。
通过加入这个正则化项，权重衰减鼓励模型的权重参数趋向于较小的值，从而提供了一种约束模型复杂度的机制。当训练神经网络时，权重衰减可以帮助防止模型对训练数据过拟合，提高其在未见过的数据上的泛化能力。
在实际应用中，可以通过在优化算法中调整正则化因子λ的大小来控制权重衰减的影响程度。较大的λ值表示更强的正则化，会导致权重参数更快地趋向于零；而较小的λ值则允许权重参数保持较大的值。
需要注意的是，权重衰减通常只应用于权重参数，而不包括偏置参数。这是因为偏置参数通常用于调整模型的输出范围，不同于权重参数对模型复杂度的贡献。

## torch.amp 混合精度
torch.amp为混合精度提供了方便的方法，其中一些操作使用torch.foat32（float）数据类型，而其他操作使用较低精度的浮点数据类型（lower_precision_fp）：torch.floot16（half）或torch.bfloat16。
一些运算，如线性层和卷积，在lower_precision_fp中要快得多。其他操作，如减少，通常需要float32的动态范围。
混合精度尝试将每个操作与其相应的数据类型相匹配。

通常，数据类型为torch.float16的“自动混合精度训练”将torch.autocast和torch.cuda.amp.GradScaler一起使用，如cuda自动混合精度示例和cuda自动组合精度配方所示。
但是，torch.autocast和torch.cuda.amp.GradScaler是模块化的，如果需要，可以单独使用。
如torch.autocast的CPU示例部分所示，数据类型为torch.bfloat16的CPU上的“自动混合精度训练/推理”仅使用torch.aautocast。

### torch.cuda.amp.grad_scaler.GradScaler
梯度scale，这正是上一小节中提到的torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow（这只是BP的时候传递梯度信息使用，真正更新权重的时候还是要把放大的梯度再unscale回去）
torch.cuda.amp.grad_scaler.GradScaler 是 PyTorch 中的一个类，用于在混合精度训练中自动缩放梯度。
它是与自动混合精度（Automatic Mixed Precision, AMP）一起使用的工具，通过调整梯度值的比例来提高数值稳定性和训练速度。
使用 GradScaler 可以简化混合精度训练中的梯度缩放过程。
以下是一个示例代码，展示了如何使用 GradScaler：

```python
import torch
from torch.cuda.amp import GradScaler

# 创建 GradScaler 对象
scaler = GradScaler()

# 前向传播和反向传播的示例代码
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    # 前向传播
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # 反向传播
    scaler.scale(loss).backward()
    
    # 梯度缩放和梯度更新
    scaler.step(optimizer)
    scaler.update()
# 在上述示例中，首先我们创建了一个 GradScaler 对象 scaler。然后，我们使用 scaler 包装了反向传播过程中的损失函数 loss，通过调用 scaler.scale(loss)，将损失值进行梯度缩放。
# 接着，我们调用 scaler.backward() 执行自动梯度计算和反向传播。
# 在梯度更新之前，我们调用 scaler.step(optimizer) 更新模型的参数，并调用 scaler.update() 更新缩放器的状态。这样，GradScaler 会根据损失值自动调整梯度的比例。
# 使用 GradScaler 可以帮助在混合精度训练中更好地处理数值不稳定性，提高训练速度和效果。
```

## torch.backends
torch.backends controls the behavior of various backends that PyTorch supports.
### torch.backends.cudnn.benchmark = True
This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

## torch.optim.lr_scheduler
torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
### torch.optim.lr_scheduler.MultiStepLR
`class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)`
更新策略：每次遇到milestones中的epoch，做一次更新

Sorry, my code level is not good.
I always run out of GPU memory after running code  one round of training.
I have two GPUs. 
- `GPU 0: NVIDIA GeForce RTX 2070 SUPER  Memory 8192MiB`
- `GPU 1: NVIDIA GeForce RTX 2070 SUPER  Memory 8192MiB`
```python
    args.batch_size = 4
    args.lr = 0.005
    args.epoch_size=0 # help='manual epoch size (will match dataset size if set to 0)'
```