from typing import Any, Dict

from transformers import AutoModelForSequenceClassification

from weasel.models.downstream_models.base_model import DownstreamBaseModel


class Transformers(DownstreamBaseModel):
    """A downstream model for sequence classification using the transformers library.

    Args:
        name: Usually the name of the model on the Hugging Face Model Hub. Passed on to the
            ``AutoModelForSequenceClassification.from_pretrained`` method.
        num_labels: Number of labels of your classification task.
    """
    def __init__(self, name: str, num_labels: int = 2):
        super().__init__()
        self.out_dim = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)

    def forward(self, kwargs):
        model_output = self.model(**kwargs)
        return model_output["logits"]

    def get_encoder_features(self, X: Dict, *args, **kwargs) -> Any:
        """
        This method returns features that will be used by Weasel's encoder network when Weasel.use_aux_input_for_encoder is True.
        Usually, they can just be the same features X, though they may (need to) be processed by the network or manually.

        args:
            X (Any): the same features that are provided to the main model in self.forward(.)
        Returns:
            Anything that is usable by the encoder network (usually just the same features/a tensor)
                as auxiliary input, besides the label matrix. E.g. a (n, d) tensor for a default MLP encoder.
        """
        return X["input_ids"]
