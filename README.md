# 说明

感谢 [ymcui/Chinese-PreTrained-XLNet](https://github.com/ymcui/Chinese-PreTrained-XLNet) 提供的中文预训练的模型。

` 训练起来， 非常慢，非常慢！！ `

## 修改的地方

- 新增 component_xlnet_data_processor.py

数据预处理文件，将训练文件转化为 tf_record 格式。

## ZC （Git Test）

- 新增 component_xlnet_multi_class_train.py

多分类组件，接入 xl-net 的模型输出，后面 可以自己添加层（一般不需要）

有些 日志错误懒得改过来了。

## 训练效果

训练的日志见：xlnet.log

```
收敛也比较慢，相比于 bert，估计是层数太多了，要慢慢学，-_-
```


## 附录

```
注意， 使用 use_bfloat16 为 true 时，出现

tensorflow.python.framework.errors_impl.NotFoundError: No registered 'Reciprocal' OpKernel for CPU devices compatible with node {{node ConstantFolding/foo/gradients/foo/Mean_1_grad/truediv_recip}} = Reciprocal[T=DT_BFLOAT16, _device="/job:localhost/replica:0/task:0/device:CPU:0"](foo/gradients/foo/Mean_1_grad/Const_1)
     (OpKernel was found, but attributes didn't match)
    .  Registered:  device='XLA_GPU'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_HALF]
  device='XLA_CPU'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_INT64, DT_HALF]
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_INT64, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_INT64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  
  stackoverflow 解释说，不支持(还以为可以使用 半精度加速，-_-)
  估计需要自己动手魔改了，转化类型
  
  [v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name]
  
```
