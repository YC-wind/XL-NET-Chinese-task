#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-08-23 17:12
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
from sklearn.externals import joblib
import os, re
from sklearn.metrics import classification_report
import xlnet
import modeling

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

config = {
    "in_1": "./data/train.tf_record",  # 第一个输入为 训练文件
    "in_2": "./data/dev.tf_record",  # 第二个输入为 验证文件
    "spiece_model_file": "./chinese_xlnet_mid_L-24_H-768_A-12/spiece.model",
    "model_config_path": "./chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json",
    "init_checkpoint": "chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt",
    # "init_checkpoint": "./bin/bert.ckpt-114000",  # 预训练bert模型
    "train_examples_len": 30000,
    "dev_examples_len": 3000,
    "num_labels": 8,
    "train_batch_size": 15,
    "dev_batch_size": 15,
    "num_train_epochs": 2,
    "eval_per_step": 500,
    "learning_rate": 1e-5,
    "use_tpu": False,
    "use_bfloat16": False,
    "dropout": 0.1,
    "dropatt": 0.1,
    "init": "normal",
    "init_std": 0.02,
    "init_range": 0.1,
    "clamp_len": -1,
    "summary_type": "last",
    "use_summ_proj": True,
    "cls_scope": None,
    "task_name": "multi_class",
    "warmup_steps": 0,
    "decay_method": "poly",
    "train_steps": 10000,
    "min_lr_ratio": 0.0,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
    "clip": 1.0,
    "lr_layer_decay_rate": 1.0,

    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./bin/",  # 保存模型路径
    "out_1": "./bin/"  # 保存模型路径
}


def dict2obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict2obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict2obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


is_training = True
FLAGS = dict2obj(config)
tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        # tf.logging.info('original name: %s', name)
        if name not in name_to_variable:
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def init_from_checkpoint(FLAGS, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.init_checkpoint is not None:
        if FLAGS.init_checkpoint.endswith("latest"):
            ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
            init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        else:
            init_checkpoint = FLAGS.init_checkpoint

        tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if FLAGS.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Log customized initialization
        tf.logging.info("**** Global Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
    return scaffold_fn


def get_input_data(input_file, seq_length, batch_size):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=3000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def create_model(FLAGS, input_ids, input_mask, segment_ids, labels, is_training=True):
    bsz_per_core = tf.shape(input_ids)[0]
    inp = tf.transpose(input_ids, [1, 0])
    seg_id = tf.transpose(segment_ids, [1, 0])
    inp_mask = tf.transpose(input_mask, [1, 0])
    label = tf.reshape(labels, [bsz_per_core])

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)
    summary = xlnet_model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

        if FLAGS.cls_scope is not None and FLAGS.cls_scope:
            cls_scope = "classification_{}".format(FLAGS.cls_scope)
        else:
            cls_scope = "classification_{}".format(FLAGS.task_name.lower())

        per_example_loss, logits = modeling.classification_loss(
            hidden=summary,
            labels=label,
            n_class=FLAGS.num_labels,
            initializer=xlnet_model.get_initializer(),
            scope=cls_scope,
            return_logits=True)

        total_loss = tf.reduce_mean(per_example_loss)

        return total_loss, per_example_loss, logits


def get_train_op(FLAGS, total_loss, grads_and_vars=None):
    global_step = tf.train.get_or_create_global_step()

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
        warmup_lr = (tf.cast(global_step, tf.float32)
                     / tf.cast(FLAGS.warmup_steps, tf.float32)
                     * FLAGS.learning_rate)
    else:
        warmup_lr = 0.0

    # decay the learning rate
    if FLAGS.decay_method == "poly":
        decay_lr = tf.train.polynomial_decay(
            FLAGS.learning_rate,
            global_step=global_step - FLAGS.warmup_steps,
            decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
            end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
    elif FLAGS.decay_method == "cos":
        decay_lr = tf.train.cosine_decay(
            FLAGS.learning_rate,
            global_step=global_step - FLAGS.warmup_steps,
            decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
            alpha=FLAGS.min_lr_ratio)
    else:
        raise ValueError(FLAGS.decay_method)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    if FLAGS.weight_decay == 0:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=FLAGS.adam_epsilon)
    elif FLAGS.weight_decay > 0 and FLAGS.num_core_per_host == 1:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            epsilon=FLAGS.adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            weight_decay_rate=FLAGS.weight_decay)
    else:
        raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                         "training so far.")

    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    if grads_and_vars is None:
        grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)

    if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
        n_layer = 0
        for i in range(len(clipped)):
            m = re.search(r"model/transformer/layer_(\d+?)/", variables[i].name)
            if not m: continue
            n_layer = max(n_layer, int(m.group(1)) + 1)

        for i in range(len(clipped)):
            for l in range(n_layer):
                if "model/transformer/layer_{}/".format(l) in variables[i].name:
                    abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - l)
                    clipped[i] *= abs_rate
                    tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
                        abs_rate, l, variables[i].name))
                    break

    train_op = optimizer.apply_gradients(
        zip(clipped, variables), global_step=global_step)

    # Manually increment `global_step` for AdamWeightDecayOptimizer
    if isinstance(optimizer, AdamWeightDecayOptimizer):
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op, learning_rate, gnorm


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.include_in_weight_decay = include_in_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        for r in self.include_in_weight_decay:
            if re.search(r, param_name) is not None:
                return True

        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    tf.logging.info('Adam WD excludes {}'.format(param_name))
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_len], name='input_ids')
input_mask = tf.placeholder(tf_float, shape=[None, FLAGS.max_seq_len], name='input_mask')
segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_len], name='segment_ids')
labels = tf.placeholder(tf.int32, shape=[None, ], name='label_ids')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

(total_loss, per_example_loss, logits) = create_model(FLAGS, input_ids, input_mask, segment_ids, labels)
train_op, learning_rate, _ = get_train_op(FLAGS, total_loss)

input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(config["in_1"], FLAGS.max_seq_len,
                                                                FLAGS.train_batch_size)

dev_batch_size = config["dev_batch_size"]

init_global = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)  # 保存最后top3模型

with tf.Session() as sess:
    sess.run(init_global)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    print("start load the pretrain model")
    scaffold_fn = None
    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
        if FLAGS.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
                return tf.train.Scaffold()


            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            # var.trainable = False
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    # if init_checkpoint:
    #     saver.restore(sess, init_checkpoint)
    #     print("checkpoint restored from %s" % init_checkpoint)
    print("********* bert_multi_class_train start *********")


    def train_step(ids, mask, segment, y, step):
        feed = {input_ids: ids,
                input_mask: mask,
                segment_ids: segment,
                labels: y}
        _, out_loss, out_logits = sess.run([train_op, total_loss, logits], feed_dict=feed)
        pre = np.argmax(out_logits, axis=-1)
        acc = np.sum(np.equal(pre, y)) / len(pre)
        print("step :{},loss :{}, acc :{}".format(step, out_loss, acc))
        return out_loss, pre, y


    def dev_step(ids, mask, segment, y):
        feed = {input_ids: ids,
                input_mask: mask,
                segment_ids: segment,
                labels: y}
        out_loss, p_ = sess.run([total_loss, logits], feed_dict=feed)
        pre = np.argmax(p_, axis=-1)
        acc = np.sum(np.equal(pre, y)) / len(pre)
        print("loss :{}, acc :{}".format(out_loss, acc))
        return out_loss, pre, y


    min_total_loss_dev = 999999
    num_train_steps = int(FLAGS.train_examples_len / config["train_batch_size"] * config["num_train_epochs"])
    num_dev_steps = int(FLAGS.dev_examples_len / config["dev_batch_size"])

    for i in range(num_train_steps):
        # batch 数据
        i += 1
        ids_train, mask_train, segment_train, y_train = sess.run([input_ids2, input_mask2, segment_ids2, labels2])
        train_step(ids_train, mask_train, segment_train, y_train, i)

        if i % FLAGS.eval_per_step == 0:
            total_loss_dev = 0
            dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2 = get_input_data(config["in_2"],
                                                                                            FLAGS.max_seq_len,
                                                                                            FLAGS.dev_batch_size)
            total_pre_dev = []
            total_true_dev = []
            for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                    [dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2])
                out_loss, pre, y = dev_step(ids_dev, mask_dev, segment_dev, y_dev)
                total_loss_dev += out_loss
                total_pre_dev.extend(pre)
                total_true_dev.extend(y_dev)
            #
            print("dev result report:")
            print(classification_report(total_true_dev, total_pre_dev))

            if total_loss_dev < min_total_loss_dev:
                print("save model:\t%f\t>%f" % (min_total_loss_dev, total_loss_dev))
                min_total_loss_dev = total_loss_dev
                saver.save(sess, config["out"] + 'bert.ckpt', global_step=i)
sess.close()

# remove dropout
print("remove dropout in predict")
tf.reset_default_graph()
is_training = False
tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_len], name='input_ids')
input_mask = tf.placeholder(tf_float, shape=[None, FLAGS.max_seq_len], name='input_mask')
segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_len], name='segment_ids')
labels = tf.placeholder(tf.int32, shape=[None, ], name='label_ids')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'
(total_loss, per_example_loss, logits) = create_model(FLAGS, input_ids, input_mask, segment_ids, labels,
                                                      is_training=False)

init_global = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存最后top3模型

try:
    checkpoint = tf.train.get_checkpoint_state(config["out"])
    input_checkpoint = checkpoint.model_checkpoint_path
    print("[INFO] input_checkpoint:", input_checkpoint)
except Exception as e:
    input_checkpoint = config["out"]
    print("[INFO] Model folder", config["out"], repr(e))

with tf.Session() as sess:
    sess.run(init_global)
    saver.restore(sess, input_checkpoint)
    saver.save(sess, config["out_1"] + 'bert.ckpt')
sess.close()

"""
********* bert_multi_class_train start *********
step :1,loss :2.11655855178833, acc :0.3333333333333333
step :2,loss :2.148242712020874, acc :0.06666666666666667
"""
