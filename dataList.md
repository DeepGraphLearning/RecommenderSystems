# Benchmark Datasets for RecSys
Here are some widely-used benchmark datasets for evaluting recommendation methods. 

## General data

| Data         | #Users    | #Items    | #Event     | #User links | Link type  | W/ Time | W/ KG |
|--------------|-----------|-----------|------------|-------------|------------|---------|-------|
| [DoubanMusic](https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0)  | 39,742    | 164,223   | 1,792,501  | 1,908,081   | 1,908,081  | Yes     |       |
| [DoubanMovie](https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0)  | 94,890    | 81,906    | 11,742,260 | 1,908,081   | 1,908,081  | Yes     |       |
| [DoubanBook](https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0)   | 46,548    | 212,995   | 1,908,081  | 1,908,081   | 1,908,081  | Yes     |       |
| Yelp\*         | 1,183,362 | 156,639   | 4,736,897  | 39,846,890  | Friendship | Yes     |       |
| [Delicious](https://grouplens.org/datasets/hetrec-2011/)    | 1867      | 40897     | 437,594    | 15,328      | Friendship | Yes     |       |
| [Last.FM1](https://grouplens.org/datasets/hetrec-2011/)     | 1892      | 12,523    | 186,480    | 25,435      | Friendship | Yes     |       |
| [MovieLens-1M](https://grouplens.org/datasets/movielens/) | 71,567    | 10,681    | 10,000,054 | -           | -          |         | [Yes](https://github.com/hwwang55/RippleNet)   |
| [FilmTrust](https://www.librec.net/datasets.html#filmtrust)    | 1,508     | 2,071     | 35,497     | 1,853       | Trust      |         |       |
| [Jester](http://www.ieor.berkeley.edu/~goldberg/jester-data/)       | 73,421    | 100       | 4,100,000  | -           | -          |         |       |
| [BookCrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) | 278,858   | 271,379   | 1,149,780  | -           | -          |         | [Yes](https://github.com/hwwang55/RippleNet)   |
| [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html)      | 107,092   | 1,280,969 | 6,442,890  | 950,327     | Friendship | Yes     |       |

* [Amazon-review](http://jmcauley.ucsd.edu/data/amazon/). It contains a large corpus of product reviews collected from Amazon.

\*: We can't find these data set(s) online anymore. If you want to use it, please feel free to contact Weiping (songweiping@pku.edu.cn).



## Session-based recommendation
* [Yoochoose](https://2015.recsyschallenge.com/challenge.html). User clicks in sessoin manner.
* Tmall. Search `User Behavior Data on Taobao/Tmall IJCAI16 Contest` on https://tianchi.aliyun.com.
* [30Music](http://recsys.deib.polimi.it/datasets/). Listening and playlists data.

## CTR Prediction
* [KDD2012]( https://www.kaggle.com/c/kddcup2012-track2).
* [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction).
* [Criteo]( https://www.kaggle.com/c/criteo-display-ad-challenge).

## KG for Recommendation
* [KB4Rec](https://github.com/RUCDM/KB4Rec). It provides linkages between movie data and Freebase.
* [MovieLens-1M](https://github.com/hwwang55/RippleNet). It merges MovieLens-1M with Microsoft Satori.
* [Book-Crossing](https://github.com/hwwang55/RippleNet). It merges Book-Crossing with Microsoft Satori.

# Acknowlegement & References:
* [Librec](https://www.librec.net/datasets.html)
