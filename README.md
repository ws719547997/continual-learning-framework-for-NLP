# continual-learning-framework-for-NLP
## 前言

关于这个持续学习框架，我暂时希望它能拥有如下的特点，这是我在科研和工作中总结出来的：

### dataloader

- 方便的选择不同的数据集，不同的任务构成一个任务序列；
- 对任务序列中任务的数量、顺序可以方便的编排与控制，方便打乱序列进行多次实验；
- 由于不同种类的任务可能对应不同的tokenizer，希望能在各种预训练语言模型之间切换；
- 我还希望能够支持多卡训练；
- 希望能够将数据集的相关信息载入日志中；
- 希望任务信息同样能够存储在dataloader中，方便在eval时选择合适的评价指标；
- clue是个傻逼吧，弄一个test_public也好啊，弄个nolabel，我评测个蛋。
- 真要命，如果是载入预料做预训练的话，是不是又产生了完全不同的逻辑，想想看怎么安排各种封装比较合理。

由于项目是基于之前持续学习情感分析的项目开发而来的，因此率先对分类任务进行支持。
#### 方便的选择不同的数据集，不同的任务构成一个任务序列
用列表作为任务序列的读取，其中的每个字符串是一个任务名，然后TaskList类会根据每一个文件名选择对应的读取方法（对于框架中不包含的数据集，这部分需要自己实现）。
在已经支持的数据集中包含两种类型：一个大数据集下有很多细分领域（例如中文的JD21和STOCK，英文的SNAP21和AMZ20），也可以是常规的单一任务数据集（例如JD70K），
只要您自己支持就行了。

#### 由于不同种类的任务可能对应不同的tokenizer，希望能在各种预训练语言模型之间切换
思考之后，实验用的预训练语言模型应该是一个不会在训练过程中来回变换，因此可以根据选择的模型自动向TaskList中传入对应的Tokenizer。
但是需要注意的是，每个任务数据的处理方式可能不同，可能分类在篇章情感分类中是：[cls] sentence，但是在基于视角的情感分类中就成了:[cls] aspect [sep] sentence
这一点需要自己在任务的loader中做具体处理。

#### 希望任务信息同样能够存储在dataloader中，方便在eval时选择合适的评价指标
同样是出于任务不同，度量指标也不同的考虑，因此特在task类中设置了变量task.task_output 用来标示输出的内容，如用数字表示是几分类，其他字段还没确定...

### 度量
对于度量的重要性是我在工作中体会到的，正如鲁迅所说：“你无法度量，也就无法改进。” 无法度量给我在工作中带来了极致蛋碎的体验。

因此我想在代码重构之初，就将度量纳入考量，其中有几个点：
- 训练过程中模型在dev和test上的测试指标
- 训练过程中时间、GPU利用率和显存
- 关于数据集的信息，比如有多少数据
- 最后能有详细的评测指标，且可以自己设定一些主流的度量方法
- 希望能看到被错误分类的样本

#### 希望能看到被错误分类的样本
我真是太愚蠢了，ptm生成的token还能被还原回来，也就是dataloader里面的batch在，数据就能恢复回来，真是太好了。