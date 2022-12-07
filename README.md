# Final project for 6.8610
This is a modified version of the OpenTag model and its read me can be found below. 

# OpenTag_2019
[Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title](https://www.aclweb.org/anthology/P19-1514)
> 该论文是在[OpenTag: Open Attribute Value Extraction from Product Profiles](https://arxiv.org/pdf/1806.01264.pdf)的基础上做的改进。模型结构如下：

(This paper is an improvement based on OpenTag: Open Attribute Value Extraction from Product Profiles. The model structure is as follows:)


![模型结构](/img/1.png)

### requirements
> 1. pytorch
> 2. pytorch-transformer
> 3. sklearn
> 4. seqeval
> 5. tqdm
> 6. torchcrf

### data
1. data目录下的raw.txt 是全量数据，中文品牌_适用季节.pkl 是从中抽出用来实验的小数据集
2. utils下的data_process.py 提供两种获得实验数据的方式，bert分词和不用bert分词,运行 
python data_process.py 可以得到 中文品牌_适用季节.pkl
3. 想要获取全量数据自己看data_process.py 应该也可以看明白了
4. dataset.py 封装了Dataset和DataLoader

- The raw.txt in the data directory is the full amount of data, and the Chinese brand_applicable season.pkl is a small data set extracted from it for experiments
- data_process.py under utils provides two ways to obtain experimental data, bert word segmentation and without bert word segmentation, run python data_process.py to get Chinese brand_applicable season.pkl
- If you want to get the full amount of data, you should be able to understand it by looking at data_process.py
- dataset.py encapsulates Dataset and DataLoader


### model
1. 提供了两个模型，LSTM_CRF.py 做一个baseline
2. OpenTag_2019.py 复现的是该论文的模型结构

- Two models are provided, LSTM_CRF.py makes a baseline
- OpenTag_2019.py reproduces the model structure of the paper

### run
1. python main.py train --batch_size=128 即可运行
2. 相应的配置可以更改config.py

- python main.py train --batch_size=128 to run
- The corresponding configuration can change config.py

### result
> 1. 没有很仔细的去调参，该结果看看就好了。需要注意的是使用bert时，lr应该在2e-5、3e-5等，bert对学习率还是非常敏感的
> 2. 在小量数据集上的实验结果 (中文品牌_适用季节.pkl)

> - I didn't tune the parameters very carefully, just look at the results. It should be noted that when using bert, lr should be 2e-5, 3e-5, etc. bert is still very sensitive to the learning rate
> - Experimental results on a small amount of data set (Chinese brand_applicable season.pkl)

+ LSTM_CRF
![lstm_crf](/img/2.png)

+ OpenTag_2019
![opentag_2019](/img/3.png)
