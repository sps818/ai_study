### `RNN` 存在的致命缺陷：
- 1. 依赖循环，每个时间步之间都是严格时序的！无法并行！速度很慢！
  - 无法通过分布式高性能并行！
  - 无法合理使用硬件来并行加速！
  - 太慢！
- 2. 存在长距离依赖问题！序列太长，容易遗忘前面的信息！
  - 手递手传递，会遗忘前面的信息！
  - 太短！
- 3. 多层堆叠无法获得足够的性能回报！
  - 太小！

### `Transformer`点对点的解决了`RNN`的问题
- 定位：
  - Transformer 聚集了长达10年NLP+CV的所有优秀成果！
  - 是前人点点滴滴贡献的集大成者！
  - Attention is all you need!
  - You Only Look Once!
  - 原地死亡！

- 特点：
  - 1. 干掉循环！干掉RNN！干掉卷积！
    - 可以通过硬件进行高性能并行计算了！
    - RNN是通过训练抽取时序特征的，Transformer计算特征时是并行的，那么，它是如何提取时序特征的呢？
    - 速度快！
  - 2. 并行计算，记忆更长久，适合处理长序列！
    - 很长！
  - 3. 可以通过堆叠获取更高的性能！
    - Scaling Law 参数量越大，数据越多，性能越强！
    - ResBlock
        - 防止梯度消失，更好的收敛
    - LayerNorm  <-- BatchNorm <-- StardardScaler
        - 防止梯度爆炸，更好的收敛

### `Transformer`的缺点：
- 计算代价太高！需要借助大量的显存！

### `Seq2Seq` VS `Transformer`
- 1. 他们俩的关系就是"油车"和"电车"的关系！
  - 外部特性：一模一样！ --> 自回归式生成算法！
  - 内部特征：本质区别！ --> 特征抽取方式不同！
- 2. Seq2Seq：以 RNN 为核心的 Encoder-Decoder 模型！
- 3. Transformer：以 Self-Attention 为核心的 Encoder-Decoder 模型！
- 4. 先内化，再泛化！

### `Transformer` 的学习：
- 1. 论文：Attention is all you need
  - https://arxiv.org/abs/1706.03762
- 2. 博客1：图解版 transformer
  - https://jalammar.github.io/illustrated-transformer/
- 3. 博客2：代码版 transformer 
  - 


