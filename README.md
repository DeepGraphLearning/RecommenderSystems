# A library of Recommender Systems
This repository provides a summary of our research on Recommender Systems.
It includes our code base on different recommendation topics, a comprehensive 
reading list and a set of bechmark data sets.

## Code Base
Currently, we are interested in sequential recommendation, feature-based 
recommendation and social recommendation.

### *Sequential Recommedation*
Since users' interests are naturally dynamic, modeling users' sequential behaviors 
can learn contextual representations of users' current interests and therefore provide 
more accurate recommendations. In this project, we include some state-of-the-art 
sequential recommenders that empoly advanced sequence modeling techniques, such as 
Markov Chains (MCs), Recurrent Neural Networks (RNNs), Temporal Convolutional Neural
Networks (TCN) and Self-attentive Neural Networks (Transformer). 

### *Feature-based Recommendation*
A general method for recommendation is to predict the click probabilities given 
users' profiles and items' features, which is known as CTR prediction.
For CTR prediction, a core task is
to learn (high-order) feature interactions because feature combinations are usually
powerful indicators for prediction. However, enumerating all the possible high-order 
features will exponentially increase the dimension of data, leading to a more serious 
problem of model overfitting. In this work, we propose to learn low-dimentional 
representations of combinatorial features with self-attention mechanism, by which 
feature interactions are automatically implemented. Quantitative results show that 
our model have good prediction performance as well as satisfactory efficiency.

### *Social recommendation*
Online social communities are an essential part of today's online experience. What we do
or what we choose may be explicitly or implicitly influenced by our friends.
In this project, we study the social influences in session-based recommendations, which 
simultaneously model users' dynamic interests and context-dependent social influences.
First, we model users' dynamic interests with recurrent neural networks. 
In order to model context-dependent social influences, we propose to employ attention-based
graph convolutional neural networks to differentiate friends' dynamic infuences in different 
behavior sessions.

## Reading List
We maintain a reading list of RecSys papers to keep track of up-to-date research.

## Data List
We provide a summary of existing benchmark data sets for evaluating recommendation methods.

## New Data set
We contribute a new large-scale dataset, which is collected from a popular movie/music/book review website Douban (www.douban.com).
The data set could be useful for researches on sequential recommendation, social recommendation and multi-domain recommendation.
See details [here](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data).


## Publications:
* Weiping Song, Zhijian Duan, Ziqing Yang, Hao Zhu, Ming Zhang and Jian Tang. [Explainable Knowledge Graph-based Recommendation via Deep Reinforcement Learning](https://arxiv.org/pdf/1906.09506.pdf). arXiv'2019.
* Weiping Song, Zhiping Xiao, Yifan Wang, Laurent Charlin, Ming Zhang and Jian Tang. 
[Session-based Social Recommendation via Dynamic Graph Attention Networks](https://arxiv.org/pdf/1902.09362.pdf). WSDM'19.
* Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang and Jian Tang.
[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf).
CIKM'2019.
