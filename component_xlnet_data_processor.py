#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-08-23 16:06
"""
import unicodedata
import six
import sentencepiece as spm
import tensorflow as tf
import collections
import pandas as pd
from sklearn.utils import shuffle

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2
    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


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


def convert_single_example(text_a, text_b, max_seq_length, tokenize_fn):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenize_fn(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenize_fn(text_b)
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for two [SEP] & one [CLS] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]

    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(SEG_ID_B)
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_B)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens

    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return (input_ids, input_mask, segment_ids)


def file_based_convert_examples_to_features(path, label2id, max_seq_length, tokenize_fn, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    tf.logging.info("Create new tfrecord {}.".format(output_file))
    writer = tf.python_io.TFRecordWriter(output_file)
    df = pd.read_csv(path, index_col=0)
    df = shuffle(df)
    count = 0
    for index, row in df.iterrows():
        # label = label2id[row["topic"].strip()]
        feature = convert_single_example(row[config["column_name_x1"]],
                                         row[config["column_name_x2"]] if config["column_name_x2"] != "" else None,
                                         max_seq_length, tokenize_fn)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        label = label2id.get(str(row[config["column_name_y"]]))
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature[0])
        features["input_mask"] = create_float_feature(feature[1])
        features["segment_ids"] = create_int_feature(feature[2])
        features["label_ids"] = create_int_feature([label])
        count += 1
        if count < 5:
            print("*** Example ***")
            print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))

            print("label: %s (id = %s)" % (row[config["column_name_y"]], str(label)))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        if count % 1000 == 0:
            print(count)
    writer.close()
    print("example count:", count)


label2id = {'劳动纠纷': 0, '婚姻家庭': 1, '公司法': 2, '交通事故': 3, '合同纠纷': 4, '刑事辩护': 5, '房产纠纷': 6, '债权债务': 7}
config = {
    "spiece_model_file": "./chinese_xlnet_mid_L-24_H-768_A-12/spiece.model",
    "csv_file": "./data/dev.csv",
    "tf_record_file": "./data/dev.tf_record",
    "column_name_x1": "question",
    "column_name_x2": "",
    "column_name_y": "label",
    "max_seq_len": 128,
}

SPIECE_UNDERLINE = '▁'
sp = spm.SentencePieceProcessor()
sp.Load(config["spiece_model_file"])


def tokenize_fn(text):
    text = preprocess_text(text, lower=False)
    return encode_ids(sp, text)


file_based_convert_examples_to_features(config["csv_file"], label2id, config["max_seq_len"], tokenize_fn,
                                        config["tf_record_file"])
