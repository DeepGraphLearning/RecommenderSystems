# Social Recommendation
Online social communities are an essential part of today's online experience. Therefore, social recommendation has drawn extensive attention in recent years. 
We argue that users' choices are determined by two main factors, dynamic interests and social influences. Based on this assumption, we develop a session-based social recommender system called DGRec in this project.
Specifically, we model users' dynamic interest using recurrent neural networks, and model the context-dependent social influences with dynamic graph attention networks. Please refer to our WSDM'19 paper [Session-based Social Recommendation via Dynamic Graph Attention Networks](https://dl.acm.org/citation.cfm?id=3290989) for further details.


Next, we introduce how to run our model for provided example data or your own data.


## Environment
* Python 2.7
* TensorFlow 1.4.1
* Pandas 0.23.3
* Numpy 1.15.0

## Usage
As an illustration, we provide the data and running command for DoubanMovie.

### Input data:
Run ```tar -xzvf data.tar.gz``` to get the processed DoubanMovie data, which contain following files:
- train.tsv: includes user historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId  ItemId Timestamps TimeId).
- valid.tsv: the same format as train.tsv, used for tuning hyperparameters.
- test.tsv: the same format as test.tsv, used for testing model.
- adj.tsv: includes links between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
- latest_session.tsv: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
- user_id_map.tsv: maps original string user id to int.
- item_id_map.tsv: maps original string item id to int.

Or, you can download the raw DoubanMovie dataset via the link below, and then run the preprocess_DoubanMovie.py file to generate input files from scratch. For you own data, you can organize the data in the format of raw DoubanMovie data, and then process the data using our script.


### Running the code:
After generating the required input files, Run the provided script directly:
```
sh run_movie.sh
```

## Douban data
In this project, we also provide a new large-scale dataset for recommendation research.
We collected users' ratings in three domains(i.e., movie, book and music) from Douban(www.douban.com), which is a popular review website in China.
The statistics of Douban datasets are summarized as follows:

| Dataset     | #user  | #item   | #event     |
|-------------|--------|---------|------------|
| DoubanMovie | 94,890 | 81,906  | 11,742,260 |
| DoubanMusic | 39,742 | 164,223 | 1,792,501  |
| DoubanBook  | 46,548 | 212,995 | 1,908,081  |

Besides rating data, we also crawled the social connections between users.

|           | #node   | #edge     |
|-----------|---------|-----------|
| SocialNet | 695,800 | 1,758,302 |

This new dataset can support various kinds of research on recommender systems, such as ***social recommendation***, ***dynamic recommendation*** and ***multi-domain recommendation***. 

Download: [Douban(112M)](https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0)


## Contact
Weiping Song, songweiping@pku.edu.cn

## Citation
If you use DGRec or Douban datasets in your research, please cite our paper:
```
@inproceedings{song2019session,
  title={Session-Based Social Recommendation via Dynamic Graph Attention Networks},
  author={Song, Weiping and Xiao, Zhiping and Wang, Yifan and Charlin, Laurent and Zhang, Ming and Tang, Jian},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={555--563},
  year={2019},
  organization={ACM}
}
```


## Acknowledgement
We owe William L. Hamilton many thanks for his excellent project [GraphSAGE](https://github.com/williamleif/GraphSAGE).


