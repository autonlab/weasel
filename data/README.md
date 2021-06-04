The smallest dataset, Bias in Bios is already included here in the source code/repository.
All the rest can be downloaded from this [Google drive link](https://drive.google.com/drive/folders/1v7IzA3Ab5zDEsRpLBWmJnXo5841tSOlh?usp=sharing).
Just replace this directory with the downloaded data/ directory (or move its files to this one).

For convenience, all the required data has been saved in the *.npz files.

Note that the models output 0 for the negative label (and 1 for the positive),
as such, the ground truth test & validation labels, as well as label matrix, already conform to this mapping (-1 is abstain).

You can read these files as follows:
    
    data = np.load('/data/*.npz')  # where * is any file in data/
    label_matrix = data['L']  # weak source votes
    Xtrain = data['Xtrain']  # features for training on soft labels
    Xtest = data['Xtest']  # features for evaluating the model
    Ytest = data['Ytest']  # gold labels for evaluating the model

You can split the gold dataset, into test and validation DownstreamDP datasets as follows:
            
    from utils.data_loading import split_dataset, DownstreamDP
    valid, test = split_dataset(<valid_set_size>, DownstreamDP, Xtest, Ytest)


The Spouses data was downloaded via [Snorkel tutorials](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spouse)
and left as-is, if you use it, please cite as below.

The LabelMe data was downloaded from [this link](http://fprodrigues.com//deep_LabelMe.tar.gz) (will start downloading directly).
and left as-is, if you use it, please cite as below.

# References

## Datasets
If you use the data above in your own projects, please cite the original dataset sources:

Amazon reviews:

    @inproceedings{Amazon,
      title={Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering},
      author={He, Ruining and McAuley, Julian},
      booktitle={Proceedings of the 25th International Conference on World Wide Web},
      pages={507--517},
      year={2016}
    }
    
BiasBios (professor_teacher):

    @inproceedings{biasBios,
      title={Bias in bios: A case study of semantic representation bias in a high-stakes setting},
      author={De-Arteaga, Maria and Romanov, Alexey and Wallach, Hanna and Chayes, Jennifer and Borgs, Christian and Chouldechova, Alexandra and Geyik, Sahin and Kenthapadi, Krishnaram and Kalai, Adam Tauman},
      booktitle={Proceedings of the Conference on Fairness, Accountability, and Transparency},
      pages={120--128},
      year={2019}
    }
IMDB:

    @inproceedings{IMDB,
        title = "Learning Word Vectors for Sentiment Analysis",
        author = "Maas, Andrew L.  and Daly, Raymond E.  and Pham, Peter T.  and Huang, Dan  and Ng, Andrew Y.  and Potts, Christopher",
        booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2011",
        address = "Portland, Oregon, USA",
        publisher = "Association for Computational Linguistics",
        pages = "142--150",
    }


Spouses:

    @article{Spouses,
        title={What do a million news articles look like?}, 
        author={Corney, David and Albakour, Dyaa and Martinez-Alvarez, Miguel and Moussa, Samir},
        year={2016},
        journal={Workshop on Recent Trends in News Information Retrieval},
        pages={42--47},
    }
    

LabelMe:

    @inproceedings{rodrigues2018deep,
      title={Deep learning from crowds},
      author={Rodrigues, Filipe and Pereira, Francisco},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={32},
      number={1},
      year={2018}
    }


## Weak supervision sources
If you use the Spouses labeling functions, please cite their creators from Snorkel:

    @article{Snorkel,
        author = {Ratner, Alexander and Bach, Stephen and Ehrenberg, Henry and Fries, Jason and Wu, Sen and Ré, Christopher},
        year = {2019},
        month = {07},
        title = {Snorkel: rapid training data creation with weak supervision},
        volume = {29},
        journal = {The VLDB Journal},
        doi = {10.1007/s00778-019-00552-1}
    }
    
The 12 LFs version for the IMDB dataset were inspired by the ones reported in:

    @article{TripletsMean,
        author = {Chen, Mayee F. and Cohen-Wang, Benjamin and Mussmann, Steve and Sala, Frederic and Ré, Christopher},
        year = {2021},
        title = {Comparing the Value of Labeled and Unlabeled Data in Method-of-Moments Latent Variable Estimation.},
        journal = {AISTATS}
    }
    
- All the rest are created by us.  


