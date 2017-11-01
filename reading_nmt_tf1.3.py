from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import numpy as np
import os
import datetime
import pickle
import random


file_data = "./data/sentence_full.tsv"
file_model = "./model/model.ckpt"
file_dic_eng = "./model/dic_eng.bin"
file_rdic_eng = "./model/rdic_eng.bin"
file_dic_kor = "./model/dic_kor.bin"
file_rdic_kor = "./model/rdic_kor.bin"
file_data_list = "./model/data_list.bin"
file_data_idx_list = "./model/data_idx_list.bin"
file_max_len = "./model/data_max_len.bin"
dir_summary = "./model/summary/"

pre_trained = 2
my_device = "/gpu:2"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if not os.path.exists(dir_summary):
    os.makedirs(dir_summary)


if pre_trained == 0:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)

    print("Load data file & make vocabulary...")

    data_list = []
    total_eng = ""
    total_kor = ""
    with open(file_data, "r", encoding="utf8") as tsv:
        for line in tsv:
            sep = line.split("\t")

            category = int(sep[0].replace("\ufeff", ""))
            sentence_english = sep[1].lower()
            sentence_english = sentence_english.replace("\n", "")
            sentence_korean = sep[2].lower()
            sentence_korean = sentence_korean.replace("\n", "")

            total_eng += sentence_english
            total_kor += sentence_korean
            data_list.append([list(sentence_english), list(sentence_korean), category])

    print("data_list example")
    print(data_list[0])
    print("data_list size = %d" % len(data_list))

    # make english char vocab
    symbols_eng = ["<PAD>", "<UNK>"]
    dic_eng = symbols_eng + list(set(total_eng))
    rdic_eng = {w: i for i, w in enumerate(dic_eng)}
    voc_size_eng = len(dic_eng)
    print("voc_size_eng size = %d" % voc_size_eng)
    print(rdic_eng)

    # make korean char vocab
    symbols_kor = ["<PAD>", "<UNK>", "<GO>"]
    dic_kor = symbols_kor + list(set(total_kor))
    rdic_kor = {w: i for i, w in enumerate(dic_kor)}
    voc_size_kor = len(dic_kor)
    print("voc_size_kor size = %d" % voc_size_kor)
    print(rdic_kor)

    data_idx_list = []
    eng_len_list = []
    kor_len_list = []
    for english, korean, category in data_list:
        idx_eng = []
        for eng in english:
            e = ""
            if eng in dic_eng:
                e = rdic_eng[eng]
            else:
                e = rdic_eng["<UNK>"]
            idx_eng.append(e)

        idx_kor = []
        for kor in korean:
            k = ""
            if kor in dic_kor:
                k = rdic_kor[kor]
            else:
                k = rdic_kor["<UNK>"]
            idx_kor.append(k)

        data_idx_list.append([idx_eng, idx_kor, category])
        eng_len_list.append(len(english))
        kor_len_list.append(len(korean))

    max_eng_len = max(eng_len_list)
    max_kor_len = max(kor_len_list)
    max_len = [max_eng_len, max_kor_len]
    print("max_eng_len = %d" % max_eng_len)
    print("max_kor_len = %d" % max_kor_len)
    print()

    # save dictionary
    with open(file_data_list, 'wb') as handle:
        pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list, 'wb') as handle:
        pickle.dump(data_idx_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic_eng, 'wb') as handle:
        pickle.dump(dic_eng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic_eng, 'wb') as handle:
        pickle.dump(rdic_eng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic_kor, 'wb') as handle:
        pickle.dump(dic_kor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic_kor, 'wb') as handle:
        pickle.dump(rdic_kor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_max_len, 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Load vocabulary from model file...")

    with open(file_data_list, 'rb') as handle:
        data_list = pickle.load(handle)
    with open(file_data_idx_list, 'rb') as handle:
        data_idx_list = pickle.load(handle)
    with open(file_rdic_eng, 'rb') as handle:
        rdic_eng = pickle.load(handle)
    with open(file_dic_eng, 'rb') as handle:
        dic_eng = pickle.load(handle)
    with open(file_rdic_kor, 'rb') as handle:
        rdic_kor = pickle.load(handle)
    with open(file_dic_kor, 'rb') as handle:
        dic_kor = pickle.load(handle)
    with open(file_max_len, 'rb') as handle:
        max_len = pickle.load(handle)

    print("data_list example")
    print(data_list[0])
    print("data_list size = %d" % len(data_list))

    voc_size_eng = len(dic_eng)
    print("voc_size_eng size = %d" % voc_size_eng)
    voc_size_kor = len(dic_kor)
    print("voc_size_kor size = %d" % voc_size_kor)

    max_eng_len = max_len[0]
    max_kor_len = max_len[1]
    print("max_eng_len = %d" % max_eng_len)
    print("max_kor_len = %d" % max_kor_len)
    print()


padded_eng_len = max_eng_len + 1
padded_kor_len = max_kor_len + 2
print("padded_eng_len = %d" % padded_eng_len)
print("padded_kor_len = %d" % padded_kor_len)

# split data set
SIZE_TEST_DATA = 100
random.shuffle(data_idx_list)
data_idx_list_test = data_idx_list[:SIZE_TEST_DATA]
data_idx_list = data_idx_list[SIZE_TEST_DATA:]
print("dataset for train = %d" % len(data_idx_list))
print("dataset for test = %d" % len(data_idx_list_test))
print()


'''''''''''''''''''''''''''''''''''''''''''''
BATCH GENERATOR
'''''''''''''''''''''''''''''''''''''''''''''
def generate_batch(size):
    assert size <= len(data_idx_list)

    data_x = np.zeros((size, padded_eng_len), dtype=np.int)
    data_y = np.zeros((size, padded_kor_len), dtype=np.int)
    data_t = np.zeros((size, padded_kor_len), dtype=np.int)
    len_x = np.zeros(size, dtype=np.int)
    len_y = np.zeros(size, dtype=np.int)
    len_p = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = data_idx_list[idx][0]
        len_x[a] = len(x)

        y = data_idx_list[idx][1]
        len_y[a] = len(y)
        len_p[a] = padded_kor_len

        t = data_idx_list[idx][1]
        assert len(x) > 0
        assert len(y) > 0

        x = x + [rdic_eng["<PAD>"]] * (padded_eng_len - len(x))
        y = [rdic_kor["<GO>"]] + y + [rdic_kor["<PAD>"]] * (padded_kor_len - len(y) - 1)
        t = t + [rdic_kor["<PAD>"]] * (padded_kor_len - len(t))
        assert len(x) == padded_eng_len
        assert len(y) == padded_kor_len
        assert len(t) == padded_kor_len
        assert y[-1] == rdic_kor["<PAD>"]
        assert t[-1] == rdic_kor["<PAD>"]

        data_x[a] = x
        data_y[a] = y
        data_t[a] = t

    return data_x, data_y, data_t, len_x, len_y, len_p


def generate_test_batch(size):
    assert size <= len(data_idx_list_test)

    data_x = np.zeros((size, padded_eng_len), dtype=np.int)
    data_y = np.zeros((size, padded_kor_len), dtype=np.int)
    data_t = np.zeros((size, padded_kor_len), dtype=np.int)
    len_x = np.zeros(size, dtype=np.int)
    len_y = np.zeros(size, dtype=np.int)
    len_p = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_test)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = data_idx_list_test[idx][0]
        len_x[a] = len(x)

        y = data_idx_list_test[idx][1]
        len_y[a] = len(y)
        len_p[a] = padded_kor_len

        t = data_idx_list_test[idx][1]
        assert len(x) > 0
        assert len(y) > 0

        x = x + [rdic_eng["<PAD>"]] * (padded_eng_len - len(x))
        y = [rdic_kor["<GO>"]] + y + [rdic_kor["<PAD>"]] * (padded_kor_len - len(y) - 1)
        t = t + [rdic_kor["<PAD>"]] * (padded_kor_len - len(t))
        assert len(x) == padded_eng_len
        assert len(y) == padded_kor_len
        assert len(t) == padded_kor_len
        assert y[-1] == rdic_kor["<PAD>"]
        assert t[-1] == rdic_kor["<PAD>"]

        data_x[a] = x
        data_y[a] = y
        data_t[a] = t

    return data_x, data_y, data_t, len_x, len_y, len_p


with tf.Graph().as_default():

    '''''''''''''''''''''''''''''''''''''''''''''
    BUILD NETWORK
    '''''''''''''''''''''''''''''''''''''''''''''
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Build Graph...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        with tf.device(my_device):
            SIZE_EMBED_DIM = 100
            SIZE_RNN_LAYER = 2
            SIZE_RNN_STATE = 60
            SIZE_ATTN = 60
            LEARNING_RATE = 0.001

            with tf.name_scope("input_placeholders"):
                enc_input = tf.placeholder(tf.int32, shape=[None, None], name="enc_input")
                enc_seq_len = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len")
                dec_input = tf.placeholder(tf.int32, shape=[None, None], name="dec_input")
                dec_seq_len = tf.placeholder(tf.int32, shape=[None, ], name="dec_seq_len")
                dec_pad_len = tf.placeholder(tf.int32, shape=[None, ], name="dec_pad_len")
                targets = tf.placeholder(tf.int32, shape=[None, None], name="targets")
                batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
                keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            with tf.name_scope("word_embedding"):
                embeddings_eng = tf.get_variable("embeddings_eng", [voc_size_eng, SIZE_EMBED_DIM])
                embed_enc = tf.nn.embedding_lookup(embeddings_eng, enc_input, name="embed_enc")
                embeddings_kor = tf.get_variable("embeddings_kor", [voc_size_kor, SIZE_EMBED_DIM])
                embed_dec = tf.nn.embedding_lookup(embeddings_kor, dec_input, name="embed_dec")

            with tf.variable_scope("encoder_layer"):
                cell_encode = []
                for a in range(SIZE_RNN_LAYER):
                    cell = rnn.BasicLSTMCell(SIZE_RNN_STATE)
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                    cell_encode.append(cell)
                multi_rnn_encode = rnn.MultiRNNCell(cell_encode, state_is_tuple=True)
                output_enc, state_enc = tf.nn.dynamic_rnn(multi_rnn_encode, embed_enc, sequence_length=enc_seq_len,
                                                          dtype=tf.float32)

            with tf.variable_scope("decoder_attention"):
                attn_luong = tf.contrib.seq2seq.LuongAttention(
                    num_units=SIZE_ATTN,
                    memory=output_enc,
                    memory_sequence_length=enc_seq_len,
                    name="attention_luong")

                cell_decode = []
                for a in range(SIZE_RNN_LAYER):
                    cell = rnn.BasicLSTMCell(SIZE_RNN_STATE)
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                    cell_decode.append(cell)
                multi_rnn_decode = rnn.MultiRNNCell(cell_decode, state_is_tuple=True)

                dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=multi_rnn_decode,
                    attention_mechanism=attn_luong,
                    attention_layer_size=SIZE_ATTN,
                    name="attention_wrapper")

                initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
                initial_state = initial_state.clone(cell_state=state_enc)

                output_layer = Dense(voc_size_kor, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            # train mode
            with tf.variable_scope("decoder_layer"):
                train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embed_dec,
                                                                 sequence_length=dec_pad_len,
                                                                 time_major=False)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, train_helper, initial_state, output_layer)

                output_train_dec, state_train_dec, len_train_dec = tf.contrib.seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=padded_kor_len)

            # predict mode
            with tf.variable_scope("decoder_layer", reuse=True):
                start_tokens = tf.tile(tf.constant([rdic_kor["<GO>"]], dtype=tf.int32), [batch_size], name="start_tokens")
                end_token = rdic_kor["<PAD>"]

                test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embeddings_kor,
                                                                       start_tokens=start_tokens,
                                                                       end_token=end_token)
                test_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, test_helper, initial_state, output_layer)

                output_test_dec, state_test_dec, len_test_dec = tf.contrib.seq2seq.dynamic_decode(
                    decoder=test_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=padded_kor_len)

            with tf.name_scope("train_optimization"):
                train_logits = tf.identity(output_train_dec.rnn_output, name="train_logits")
                masks = tf.sequence_mask(dec_seq_len + 1, padded_kor_len, dtype=tf.float32, name="masks")

                loss = tf.contrib.seq2seq.sequence_loss(
                    logits=train_logits,
                    targets=targets,
                    weights=masks,
                    name="loss")
                train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

            with tf.name_scope("prediction"):
                test_prediction = tf.identity(output_test_dec.sample_id, name="test_prediction")

        loss_summary = tf.summary.scalar("loss", loss)
        merged_summary = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(dir_summary, sess.graph)

        '''''''''''''''''''''''''''''''''''''''''''''
        TRAIN PHASE
        '''''''''''''''''''''''''''''''''''''''''''''
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if pre_trained == 2:
            print("Restore model file...")
            saver.restore(sess, file_model)

        BATCHS = 150
        BATCHS_TEST = 10
        EPOCHS = 500
        STEPS = int(len(data_idx_list) / BATCHS)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Train start!!")

        glob_step = 0
        for epoch in range(EPOCHS):
            for step in range(STEPS):
                data_x, data_y, dat_t, len_x, len_y, len_p = generate_batch(BATCHS)

                feed_dict = {
                    enc_input: data_x,
                    enc_seq_len: len_x,
                    dec_input: data_y,
                    dec_seq_len: len_y,
                    dec_pad_len: len_p,
                    targets: data_t,
                    batch_size: BATCHS,
                    keep_prob: 0.75
                }

                _, batch_loss, batch_loss_summ = sess.run([train_op, loss, merged_summary], feed_dict)

                if glob_step % 10 == 0:
                    train_summary_writer.add_summary(batch_loss_summ, glob_step)

                if glob_step % 50 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] glob_step[%05d] - batch_loss:%.5f  (%s)" % (epoch, glob_step, batch_loss, now))

                    saver.save(sess, file_model)

                # simple test
                if glob_step % 500 == 0:
                    data_x, data_y, data_t, len_x, len_y, len_p = generate_test_batch(BATCHS_TEST)
                    feed_dict = {
                        enc_input: data_x,
                        enc_seq_len: len_x,
                        dec_input: data_y,
                        dec_seq_len: len_y,
                        dec_pad_len: len_p,
                        targets: data_t,
                        batch_size: BATCHS_TEST,
                        keep_prob: 1.0
                    }

                    prediction_train = sess.run([train_logits], feed_dict)
                    prediction_train = prediction_train[0]
                    prediction_train = np.argmax(prediction_train, axis=2)

                    prediction_test = sess.run([test_prediction], feed_dict)
                    prediction_test = prediction_test[0]
                    print()
                    print("-" * 60)
                    for a in range(len(data_x)):
                        sen_english = [dic_eng[r] for r in data_x[a]]
                        sen_english = "".join(sen_english)

                        sen_korean = [dic_kor[s] for s in data_y[a]]
                        sen_korean = "".join(sen_korean)

                        sen_train = [dic_kor[p] for p in prediction_train[a]]
                        sen_train = "".join(sen_train)

                        sen_test = [dic_kor[p] for p in prediction_test[a]]
                        sen_test = "".join(sen_test)

                        print("test[%03d] ENG_INPUT  : %s" % (a, sen_english))
                        print("test[%03d] KOR_OUTPUT : %s" % (a, sen_korean))
                        #print("test[%03d] KOR_TRAIN : %s" % (a, sen_train))
                        print("test[%03d] KOR_PREDICT: %s" % (a, sen_test))
                        print()
                    print("-" * 60)
                    print()

                glob_step += 1

        print("Train finished!!")
        print()
