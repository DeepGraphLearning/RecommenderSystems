# Sequential Recommenders
Modeling users' dynamic interests has drawn increasingly attention. In this project, we try to implement two kinds of sequential recommenders, i.e., Markov Chains-based methods and neural network-based methods. Currently, we include following models:

## Neural models:
The task of sequential recommendation is to predict next item(s) that a user is likely to choose based on his/her ordered behavior sequence. Therefore, neural networks that are sequence-aware can be used to encode the user's history.

    * RNN [1]. This is a modified version of GRU4Rec. We train the RNN(LSTM) model via standard BBTT, instead of the per-step training in original paper[1].
    * TCN [2]. Use temporal convolutional neural network (a.k.a. dilated causal convolutional neural network) to model sequence data.
    * Transformer [3]. Use Transformer (a.k.a. self-attention neural network) to model sequence data.

## Markov Chains:
Methods which rely on Markov Chains (MCs) also assume users' next item depend on their previous item(s). We implement two main MCs-based methods.

    * FPMC [4]. FPMC models users's dynamic interests by factorizaing item-to-item transitions.
    * Fossil [5]. Fossil follows the similar idea of FPMC in modeling users' dynamic interests.

DISCLAIMER: Since we intend to unify these methods into the same framework, we cannot guarantee that all models are implemented the same as authors' official implementations.

# References
1. Hidasi et al. [Session-based recommendations with recurrent neural networks](https://arxiv.org/pdf/1511.06939.pdf). ICLR'16.
2. Yuan et al. [A Simple Convolutional Generative Network for Next Item Recommendation](https://fajieyuan.github.io/papers/nextitnet6.pdf). WSDM'19.
3. Kang and McAuley. [Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf). ICDM'18.
4. Rendle et al. [Factorizing Personalized Markov Chains for Next-Basket Recommendation](http://www.ra.ethz.ch/cdstore/www2010/www/p811.pdf). WWW'10.
5. He et al. [Fusing similarity models with markov chains for sparse sequential recommendation](https://arxiv.org/pdf/1609.09152.pdf). ICDM'16.
