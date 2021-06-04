import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_activation_function


class Encoder(nn.Module):
    """
    This encoder network is targeted at binary classification, for multi-class see the MulticlassEncoder below.
    While this version is NOT USED in any experiments, it should give very similar results for binary tasks,
    and be easier to understand, similarly as prior binary label models that are based on PGMs.
    """

    def __init__(self, input_size, dims, out_size, drop_prob=0.0, batch_norm=True, acc_scaler="sqrt",
                 accuracy_func=nn.Softmax(dim=1), act_func='relu', temperature=1, class_balance=None, cardinality=2,
                 class_conditional_accs=False):
        r"""
        :param input_size: Input dimensionality, i.e. number of labeling functions (LF).
        :param dims: a list of the hidden layer dimensions
        :param out_size: the number of encoder outputs, usually == #LFs
        :param drop_prob: dropout probability
        :param batch_norm: True or false, whether to use batch normalization.
        :param acc_scaler: How to scale the accuracies, either a float or 'sqrt', if neither of, the scaler becomes
                            input_size. This parameter can be important when the accuracy_func is softmax.
        :param accuracy_func: Eg. F.softmax or F.relu
        :param class_balance: The class balance, i.e. prior prob. on the classes. Of shape (C,).
                                If None, an uniform class balance is assumed.
        :param cardinality: The number of classes C (e.g. 2 for binary classification).
        """
        super(Encoder, self).__init__()
        if class_balance is None:
            class_balance = [1 / cardinality for _ in range(cardinality)]
        self.drop_prob = drop_prob
        self.input_size = input_size
        self.batch_norm = batch_norm
        self.cardinality = cardinality
        self.nlf = out_size  # number of labeling functions
        self.temperature = temperature
        self.class_conditional = class_conditional_accs
        activation_func = act_func
        if self.class_conditional:
            print('Using class-conditional accuracy scores')
            out_size *= self.cardinality

        # Create the network, that gets a (input_size,) tensor and outputs a (out_size,) tensor (the LF attention scores)
        encoder_modules = []
        dims = [input_size] + dims
        for i in range(1, len(dims)):
            encoder_modules.append(nn.Linear(dims[i - 1], dims[i], bias=True))
            if self.batch_norm:
                encoder_modules.append(nn.BatchNorm1d(dims[i]))
            encoder_modules.append(get_activation_function(activation_func))
            if drop_prob > 0:
                encoder_modules.append(nn.Dropout(drop_prob))

        encoder_modules.append(nn.Linear(dims[-1], out_size, bias=True))
        self.network = nn.Sequential(*encoder_modules)

        self.accuracy_func = accuracy_func
        self.p = torch.tensor(class_balance, requires_grad=False)  # class balance
        if isinstance(acc_scaler, float) or isinstance(acc_scaler, int):
            if self.accuracy_func in [nn.ReLU(), F.relu] and acc_scaler != 1.0:
                print("For ReLU no accuracy scaler is used, setting it to 1...")
                self.acc_scaler = 1.0
            else:
                assert acc_scaler > 0, 'Accuracy scaler must be positive'
                self.acc_scaler = acc_scaler
        else:
            self.acc_scaler = self.input_size
            if acc_scaler.lower() in ["sqrt", "root"]:
                self.acc_scaler = np.sqrt(self.acc_scaler)
            if self.class_conditional:
                self.acc_scaler *= self.cardinality

    def forward(self, label_matrix, extra_input=None, get_accuracies=False):
        """
        :param label_matrix: a (n, m) tensor L, where n = #samples, m = #LFs and
                                L_ij = 0 if the j-th LF abstained on i, and
                                L_ij = -1 if the j-th LF voted for the negative class
                                L_ij = +1 if the j-th LF voted for the positive class for the sample i
        :param extra_input: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :return: a (n, 1) tensor, where each entry equals the aggregated votes of the LFs (times its learned accuracy)
        """
        # get the LF attention scores/sample-dependent accuracies
        accuracies = self.get_attention_scores(label_matrix, extra_input)
        # Multiply them with their respective LF output, and sum it up for each sample
        aggregation = (accuracies * label_matrix).sum(dim=1)  # , keepdim=True).squeeze(1)
        # Map the aggregation to a probability, usually via a sigmoid or softmax
        if get_accuracies:
            return aggregation, accuracies
        return aggregation

    def get_attention_scores(self, label_matrix, extra_input=None):
        """
        :param label_matrix: a (n, m) tensor, where n = #samples, m = #LFs
        :param extra_input: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :return: a (n, m) tensor that represents the sample-dependent accuracies of the LFs
        """
        # If there's an extra input (e.g. features) we concatenate it
        input = label_matrix if extra_input is None else torch.cat((label_matrix, extra_input), dim=1)
        # Forward pass through the encoder network
        accuracies = self.network(input)  # shape (batch-size, #LFs) or (batch-size, #LFs * #classes) if class conditional accs.

        if self.class_conditional:  # make accuracies have shape (batch_size, #LFs, #classes)
            accuracies = accuracies.reshape(-1, self.nlf, self.cardinality)
        accuracies = accuracies/self.temperature
        # Produce attention scores from the encoder output
        accuracies = self.acc_scaler * self.accuracy_func(accuracies) + 1e-5  # to avoid scores=0 on init for relu

        return accuracies

    def predict_proba(self, label_matrix, extra_input=None):
        """
        :param label_matrix: a (n, m) tensor L, where n = #samples, m = #LFs and
                                L_ij = 0 if the j-th LF abstained on i, and
                                L_ij = -1 if the j-th LF voted for the negative class
                                L_ij = +1 if the j-th LF voted for the positive class for the sample i
        :param extra_input: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :return: P(y = 1| label_matrix, extra_input), a (n, 1) tensor
        """
        aggregation = self.forward(label_matrix, extra_input=extra_input, get_accuracies=False)
        return self.logits_to_probs(aggregation)

    def logits_to_probs(self, logits):
        return torch.sigmoid(logits)

    def logits_len(self):
        return 1  # single value, even though binary case


##################################################################################################################
class MulticlassEncoder(Encoder):
    """
    This is an encoder that supports multi-class classification between C classes {1, 2, ..., C}.
    We include an easier to understand binary encoder above.
    """

    def __init__(self, input_size, dims, out_size, *args, **kwargs):
        super().__init__(input_size, dims, out_size, *args, **kwargs)

    def forward(self, lf_vote_input, extra_input=None, label_matrix=None, device='cuda', get_accuracies=False):
        """
        :param label_matrix: a (n, m) tensor L, if cardinality = 2, the same representation is used as above.
                                Otherwise, in the multi-class setting, we have that
                                L_ij = 0 if the j-th LF abstained on i, and
                                L_ij = c if the j-th LF voted for class c for the sample i
        :param extra_input: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :param device: E.g. 'cuda' or 'cpu'
        :return: A (n, C) tensor, with class probabilities, if cardinality is 2, we return only a (n,1) tensor.
        """
        # get the LF attention scores/sample-dependent accuracies
        accuracies = self.get_attention_scores(lf_vote_input, extra_input)

        # Make label matrix (n, m) a one-hot/indicator matrix (n, m, C)
        L = self._create_L_ind(lf_vote_input) if label_matrix is None else label_matrix

        if self.class_conditional:
            aggregation = (L * accuracies).sum(dim=1)
            # The following snippet of code is equivalent to the (vectorized) line above:
            # aggregationLONG = torch.zeros((accuracies.shape[0], self.cardinality)).to(device)  # (n, C)
            # for i, (lf_votes, accs) in enumerate(zip(L, accuracies)):  # iterate through all n batches
                 # Multiply the class-conditional accuracies against each class indicator column
            #    aggregationLONG[i, :] = torch.sum(accs * lf_votes, dim=0)  # (m, C) * (m, C) --> (m, C) --> (1, C)
            # assert torch.allclose(aggregationLONG, aggregation), 'Not equivalent? :o'

        else:
            aggregation = (accuracies.unsqueeze(1) @ L).squeeze(1)
            # The following snippet of code is equivalent to the (vectorized) line above:
            # aggregationLONG = torch.zeros((accuracies.shape[0], self.cardinality)).to(device)  # (n, C)
            # for i, (lf_votes, accs) in enumerate(zip(L, accuracies)):  # iterate through all n batches
                  # Multiply the accuracies against each class indicator column
            #     aggregationLONG[i, :] = accs @ lf_votes  # (1, m) x (m, C) --> (1, C)
            # assert torch.allclose(aggregationLONG, aggregation), 'Not equivalent? :o'

        # Y = F.softmax(aggregation + torch.log(self.p.to(device)), dim=1)  # shape #batches x #classes
        # Y = F.softmax(aggregation, dim=1)  # shape #batches x #classes
        if get_accuracies:
            return aggregation, accuracies
        return aggregation

    def _create_L_ind(self, label_matrix):
        """ Adapted from Snorkel v0.9
        Convert a label matrix with labels in 0...k to a one-hot format.

        Parameters
        ----------
        L
            An (n,m) label matrix with values in {0,1,...,C}, where 0 means abstains

        Returns
        -------
        torch.tensor
            An (n, m, C) tensor with values in {0,1}
        """
        n, m = label_matrix.shape
        if self.cardinality == 2:
            L = label_matrix.clone()
            # In the binary case we still use the -1, +1 for the negative, positive classes respectively
            L[label_matrix == 1] = 2
            L[label_matrix == -1] = 1  # now, we mapped negative (-1) -> 1, positive (+1) --> 2
        else:
            L = label_matrix + 1
        L_ind = torch.zeros((n, m, self.cardinality), requires_grad=False)
        for class_y in range(1, self.cardinality + 1):
            # go through Y == 1 (negative), Y == 2 (positive)...
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, :, class_y - 1] = torch.where(L == class_y, 1, 0)
        return L_ind.to(L.device)

    def predict_proba(self, label_matrix, extra_input=None):
        """
        :param label_matrix: a (n, m) tensor L, where n = #samples, m = #LFs
        :param extra_input: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :return: a (n, C) tensor, where the c-th column represents P(y = c| label_matrix, extra_input).
        """
        aggregation = self.forward(label_matrix, extra_input=extra_input, get_accuracies=False)
        return self.logits_to_probs(aggregation)

    def logits_to_probs(self, logits):
        probs = F.softmax(logits, dim=1)
        return probs

    def logits_len(self):
        return self.cardinality
