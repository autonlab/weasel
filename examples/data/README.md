For illustrative purposes, we include the dataset Bias in Bios
here in the source code/repository.

Note that the models output 0 for the negative label (and 1 for the positive),
as such, the ground truth test & validation labels, as well as label matrix, already conform to this mapping (-1 is abstain).

You can read these files as follows:
    
    data = np.load('/data/*.npz')  # where * is any file in data/
    label_matrix = data['L']  # weak source votes
    Xtrain = data['Xtrain']  # features for training on soft labels
    Xtest = data['Xtest']  # features for evaluating the model
    Ytest = data['Ytest']  # gold labels for evaluating the model

# References

## Datasets
If you use the data above in your own projects, please cite the original dataset sources:

BiasBios (professor_teacher):

    @inproceedings{biasBios,
      title={Bias in bios: A case study of semantic representation bias in a high-stakes setting},
      author={De-Arteaga, Maria and Romanov, Alexey and Wallach, Hanna and Chayes, Jennifer and Borgs, Christian and Chouldechova, Alexandra and Geyik, Sahin and Kenthapadi, Krishnaram and Kalai, Adam Tauman},
      booktitle={Proceedings of the Conference on Fairness, Accountability, and Transparency},
      pages={120--128},
      year={2019}
    }