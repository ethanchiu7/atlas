# shape
- input_shape 是单个样本的shape
- batch_size 是batch的样本数量
- batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
- 以上都是tuple