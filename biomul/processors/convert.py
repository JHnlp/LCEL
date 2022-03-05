"""
=======================================================================
 Copyright (c) 2019 PolyU CBS LLT Group. All Rights Reserved

@Author      :  Jinghang GU
@Contect     :  gujinghangnlp@gmail.com
@Time        :  2021/01/01
@Description :
=======================================================================
"""
import pandas as pd
import logging
import copy
import json
import pathlib
import numpy as np

logger = logging.getLogger(__name__)

UNKOWN_LABEL = 'D000000'
UNKOWN_TERM = 'UNKOWN_TERM'


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        labels: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, labels=None, keywords=None,
                 journal=None, pub_type=None, authors=None, doi=None, mesh=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.keywords = keywords
        self.journal = journal
        self.authors = authors
        self.pub_type = pub_type
        self.doi = doi
        self.mesh = mesh

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @property
    def text(self):
        if self.text_b is not None:
            if isinstance(self.text_b, list):  # tokenized
                return self.text_a + self.text_b
            else:
                return self.text_a + ' ' + self.text_b
        return self.text_a


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        labels: Label corresponding to the input
    """

    # __slots__ = ['guid', 'input_indicators', 'attention_mask', 'token_type_indicators',
    #              'labels', 'candidate_labels', 'input_len', 'journal']

    def __init__(self, guid, input_len, input_ids, attention_mask,
                 token_type_ids, labels, keywords=None, mesh=None, journal=None):
        self.guid = guid
        self.input_len = input_len
        self.input_indicators = input_ids
        self.attention_mask = attention_mask
        self.token_type_indicators = token_type_ids
        self.labels = labels
        self.keywords = keywords
        self.mesh = mesh
        self.journal = journal

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LabelVocab(object):
    def __init__(self, filepath):
        self.stod = {}  # name string to descriptor
        self.dtos = {}  # descriptor to name string
        self.dtot = {}  # descriptor to tokenized words
        self.dtof = {}  # descriptor to frequency
        self._read_label_file(filepath)

    def __len__(self):
        return len(self.stod)

    def _read_label_file(self, filepath):
        """Reads a tab separated label file.
            File format should be: 'Name\tDescriptor\tFrequency\tTokens'
        """
        with pathlib.Path(filepath).open("r", encoding="utf8") as f:
            for ln in f:
                entry = ln.strip().split('\t')
                assert len(entry) == 4
                name, des, freq, tokens = entry
                try:
                    descriptor = int(des)
                except:
                    descriptor = des
                freq = int(freq)
                tokens = tokens.strip().split(' ')

                self.stod[name] = descriptor
                self.dtos[descriptor] = name
                self.dtot[descriptor] = tokens
                self.dtof[descriptor] = freq

    def get_descriptor_by_name(self, name):
        return self.stod.get(name, None)

    def get_name_by_descriptor(self, descriptor):
        return self.dtos.get(descriptor, None)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, file_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, file_path):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, file_path):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self, file_path):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class LitCovidProcessor(DataProcessor):
    """Processor for the Relation data set."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        super(LitCovidProcessor, self).__init__()

    def _create_examples(self, df_data):
        """Creates examples from DataFrame."""
        examples = []
        for idx, row in df_data.iterrows():
            pmid = "" if pd.isnull(row["pmid"]) else str(row["pmid"])
            title = "" if pd.isnull(row["title"]) else row["title"]
            abstract = "" if pd.isnull(row["abstract"]) else row["abstract"]
            journal = "" if pd.isnull(row["journal"]) else row["journal"]
            keywords = [] if pd.isnull(row["keywords"]) else row["keywords"].strip().split(';')
            pub_type = [] if pd.isnull(row["pub_type"]) else row["pub_type"].strip().split(';')
            authors = [] if pd.isnull(row["authors"]) else row["authors"].strip().split(';')
            doi = "" if pd.isnull(row["doi"]) else row["doi"]
            label = [] if pd.isnull(row["label"]) else row["label"].strip().split(';')
            mesh = [] if pd.isnull(row["mesh"]) else row["mesh"].strip().split(';')

            # check the entry info
            if not pmid:
                print('Empty PMID!!!')
                continue

            example = InputExample(guid=pmid, text_a=title, text_b=abstract,
                                   labels=label, keywords=keywords,
                                   journal=journal, pub_type=pub_type,
                                   authors=authors, doi=doi, mesh=mesh)
            examples.append(example)
        return examples

    def get_train_examples(self, file_path):
        file_path = pathlib.Path(file_path)
        if file_path.is_dir():
            _fp = next(file_path.glob('*Train.csv'))  # the first matching
            df_data = pd.read_csv(_fp.absolute())
        elif file_path.is_file():
            df_data = pd.read_csv(file_path)
        else:
            raise ValueError('Can not find the train file!')
        return self._create_examples(df_data)

    def get_dev_examples(self, file_path):
        file_path = pathlib.Path(file_path)
        if file_path.is_dir():
            _fp = next(file_path.glob('*Dev.csv'))  # the first matching
            df_data = pd.read_csv(_fp.absolute())
        elif file_path.is_file():
            df_data = pd.read_csv(file_path)
        else:
            raise ValueError('Can not find the dev file!')
        return self._create_examples(df_data)

    def get_test_examples(self, file_path):
        file_path = pathlib.Path(file_path)
        if file_path.is_dir():
            _fp = next(file_path.glob('*Test.csv'))  # the first matching
            df_data = pd.read_csv(_fp.absolute())
        elif file_path.is_file():
            df_data = pd.read_csv(file_path)
        else:
            raise ValueError('Can not find the test file!')
        return self._create_examples(df_data)

    def get_examples(self, data_type, file_path):
        if data_type.lower() == 'train':
            return self.get_train_examples(file_path)
        elif data_type.lower() == 'dev':
            return self.get_dev_examples(file_path)
        else:
            return self.get_test_examples(file_path)

    def get_labels(self, file_path):
        file_path = pathlib.Path(file_path)
        if file_path.is_dir():
            _fp = file_path.joinpath('label_list.txt')
        elif file_path.is_file():
            _fp = file_path
        else:
            raise ValueError('Can not find the label list file!')
        with pathlib.Path(_fp).open('r', encoding='utf8') as inf:
            label_list = inf.read().strip().split('\n')
        return label_list


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 pad_token=0,
                                 pad_on_left=False,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (exam_idx, example) in enumerate(examples, 1):
        if exam_idx % 10000 == 0:
            logger.info("--Writing example %d " % (exam_idx))

        # keywords
        _tokens_kw = [tokenizer.tokenize(kw) for kw in example.keywords]
        _tokens_kw = [tks + ['<TAG>'] for tks in _tokens_kw[:-1]] + [_tokens_kw[-1]] if len(_tokens_kw) else []
        tokens_keywords = []
        for tks in _tokens_kw:
            tokens_keywords.extend(tks)

        # MeSH
        _tokens_mesh = [tokenizer.tokenize(kw) for kw in example.mesh]
        _tokens_mesh = [tks + ['<TAG>'] for tks in _tokens_mesh[:-1]] + [_tokens_mesh[-1]] if len(_tokens_mesh) else []
        tokens_mesh = []
        for tks in _tokens_mesh:
            tokens_mesh.extend(tks)

        # journal
        _tokens_jnl = tokenizer.tokenize(example.journal)

        tokens = tokens_keywords + [sep_token] + tokens_mesh + [sep_token] + _tokens_jnl + [sep_token]

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += tokens_a + [sep_token]
        token_type_indicators = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            token_type_indicators += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if len(tokens) > max_seq_length - 1:  # sequence is too long
            tokens = tokens[:max_seq_length - 1]
            tokens[-1] = sep_token
            token_type_indicators = token_type_indicators[:max_seq_length - 1]

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            token_type_indicators = token_type_indicators + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_type_indicators = [cls_token_segment_id] + token_type_indicators

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            token_type_indicators = ([pad_token_segment_id] * padding_length) + token_type_indicators
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_indicators = token_type_indicators + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_indicators) == max_seq_length

        label_id = np.array([label_map[lb] for lb in example.labels])
        label_id_binary = np.zeros(len(label_list))
        label_id_binary[label_id] = 1

        if exam_idx < 4:
            print('\n')
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_indicators: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("token_type_indicators: %s" % " ".join([str(x) for x in token_type_indicators]))
            logger.info("labels num: %d" % (len(example.labels)))

        features.append(
            InputFeatures(guid=example.guid,
                          input_len=max_seq_length,
                          input_ids=input_ids,
                          attention_mask=input_mask,
                          token_type_ids=token_type_indicators,
                          labels=label_id_binary,
                          journal=None))

    return features


processors = {
    "litcovid": LitCovidProcessor,
}

if __name__ == '__main__':
    processor = LitCovidProcessor()
    tr = processor.get_train_examples('../../data')
    label_list = processor.get_labels('../../data')

    pass
