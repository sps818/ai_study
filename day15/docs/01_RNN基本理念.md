### 1. RNN 到底要解决什么问题？
- 时序信号的特征抽取！
- 顺序！

### 2. RNN 的解决方法：
- 循环！！！
- 逐个处理！！
- 前后依赖！！！
  - 后一项直接依赖前一项，间接依赖前面所有的项！
  - 前后进行 `手递手` 传递！
- 中间隐藏状态！！！

### 3. RNN API 如何调用？
- nn.RNN
  - 自动挡（自动循环）
  - 用于编码器
- nn.RNNCell
  - 手动挡（手动循环）
  - 用于解码器

### 4. LSTM 博客
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/


### 5. 信息：
- CRUD工程师
  - 1. 增删改查
- LSTM：
  - 1. 门控思想
  - 2. CRUD思想
    - 1. 遗忘门：先丢掉一部分不重要信息！
    - 2. 输入门：增加重要的信息！
    - 
