from typing import Tuple
import pickle
import gzip
import os

import torch
from torchtext.data import Dataset, Iterator, Field
from torchtext import data


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class HIndexDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
            self,
            path: str,
            fields: Tuple[Field, Field, Field, Field, Field],
            **kwargs
    ):

        path = [path]
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("author", fields[0]),
                ("autemb", fields[1]),
                ("papemb", fields[2]),
                ("features", fields[3]),
                ("target", fields[4])
            ]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for aut_id in tmp:
                s = tmp[aut_id]
                samples[aut_id] = {
                    "name": aut_id,
                    "autemb": s["author_embedding"],
                    "papemb": s["papers_embedding"],
                    "features": s["features"],
                    "target": s["target"],
                }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        # This is for numerical stability
                        sample["autemb"] + 1e-8,
                        sample["papemb"] + 1e-8,
                        sample["features"] + 1e-8,
                        sample["target"],
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)


class HIndexDatasetTest(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
            self,
            path: str,
            fields: Tuple[Field, Field, Field, Field, Field],
            **kwargs
    ):

        path = [path]

        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("author", fields[0]),
                ("autemb", fields[1]),
                ("papemb", fields[2]),
                ("features", fields[3]),
            ]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for aut_id in tmp:
                s = tmp[aut_id]
                samples[aut_id] = {
                    "name": aut_id,
                    "autemb": s["author_embedding"],
                    "papemb": s["papers_embedding"],
                    "features": s["features"],
                }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        # This is for numerical stability
                        sample["autemb"] + 1e-8,
                        sample["papemb"] + 1e-8,
                        sample["features"] + 1e-8,
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)


def load_data(data_cfg, training=True):
    data_path = data_cfg.model_dir + data_cfg.data_path
    train_paths = os.path.join(data_path, data_cfg.splitpaths[0])
    dev_paths = os.path.join(data_path, data_cfg.splitpaths[1])
    test_path = os.path.join(data_path, data_cfg.splitpaths[2])
    pad_feature_size = data_cfg.paper_dim

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    author_name_field = data.RawField()
    author_name_field.is_target = False

    author_embedding_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=False,
        postprocessing=stack_features
    )

    papers_embedding_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    features_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=False,
        postprocessing=stack_features
    )

    label_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        tokenize=lambda features: features,
        batch_first=True,
        postprocessing=stack_features,
    )
    if training:
        train_data = HIndexDataset(
            path=train_paths,
            fields=(author_name_field, author_embedding_field, papers_embedding_field, features_field, label_field)
        )

        dev_data = HIndexDataset(
            path=dev_paths,
            fields=(author_name_field, author_embedding_field, papers_embedding_field, features_field, label_field)
        )
        return train_data, dev_data
    else:
        test_data = HIndexDatasetTest(
            path=test_path,
            fields=(author_name_field, author_embedding_field, papers_embedding_field, features_field)
        )
        return test_data


def make_data_iter(
        dataset: Dataset,
        batch_size: int,
        train: bool = False,
        shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.
    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=False,
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
            sort_within_batch=False
        )
    return data_iter


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
            self,
            torch_batch,
            author_dim,
            paper_dim,
            is_train: bool = True,
            use_cuda: bool = True,
    ):

        # Author Information
        self.author = torch_batch.author

        # Papers
        self.papemb, self.papemb_length = torch_batch.papemb
        self.pap_dim = paper_dim
        self.papemb_mask = (self.papemb != torch.zeros(paper_dim))[..., 0].unsqueeze(1)

        # Author
        self.autemb = torch_batch.autemb
        self.aut_dim = author_dim
        self.autemb_mask = (self.autemb != torch.zeros(author_dim))[..., 0].unsqueeze(1)

        # Features
        self.features = torch_batch.features

        # Target
        if is_train:
            self.target = torch_batch.target
        else:
            self.target = None

        # Other
        self.use_cuda = use_cuda
        self.num_seqs = self.papemb.size(0)

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU
        :return:
        """
        self.papemb = self.papemb.cuda()
        self.papemb_mask = self.papemb_mask.cuda()

        self.autemb = self.autemb.cuda()
        self.autemb_mask = self.autemb_mask.cuda()

        self.features = self.features.cuda()

        if self.target is not None:
            self.target = self.target.cuda()
