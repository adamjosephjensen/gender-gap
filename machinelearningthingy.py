import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
import re
from vocab import PAD_ID, UNK_ID
import pdb  # noqa


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping
    gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence)  # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum
      length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0,
        otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad # noqa
    # return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch) # noqa
    padded = [lis + [PAD_ID] * (maxlen - len(lis)) for lis in token_batch]
    return padded


class GameModel(object):
    """
    stores the game state and makes predictions
    """

    def __init__(self, id2word, word2id, emb_matrix, rule_length=14):
        self.new_game()
        self.num_epochs = 2000
        self.sess = tf.Session()
        self.rule_length = rule_length
        self.hidden_size = 100
        self.id2word = id2word
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(),
                                                     name='KEEP_PROB')
        self.word2id = word2id
        cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(cell_fw,
                                          input_keep_prob=self.keep_prob)
        cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(cell_bw,
                                          input_keep_prob=self.keep_prob)
        # add all parts of the graph
        with tf.variable_scope("MODEL"):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # clip by gradient norm
        params = tf.trainable_variables()
        self.param_norm = tf.global_norm(params)
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        # Define optimizer and return update step
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(beta1=0.8)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                           global_step=self.global_step)
        self.summaries = tf.summary.merge_all()

    def new_game(self):
        guess_cols = ["Rule Guess", "Penalty"]
        self.guesses = pd.DataFrame(columns=guess_cols)
        list_cols = ["Sequence", "Follows Rule"]
        self.lists = pd.DataFrame(columns=list_cols, data=[[str([2, 4, 6]), True]])  # noqa

    def add_placeholders(self):
        shp = [None, self.rule_length]
        self.rule_ids = tf.placeholder(tf.int32, shape=shp, name='RULE_IDS')
        self.rule_mask = tf.placeholder(tf.int32, shape=shp, name='RULE_MASK')
        self.labels = tf.placeholder(tf.int32, shape=(None), name='LABELS')

    def add_embedding_layer(self, emb_matrix):
        with tf.variable_scope("embedding"):
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32,
                                           name="emb_matrix")
            self.rule_embs = embedding_ops.embedding_lookup(embedding_matrix,
                                                            self.rule_ids)

    def build_graph(self):
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(self.rule_mask, reduction_indices=1)
            _, finals = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw,
                                                        self.rnn_cell_bw,
                                                        self.rule_embs,
                                                        input_lens,
                                                        dtype=tf.float32)
            (output_state_fw, output_state_bw) = finals
            out = output_state_fw + output_state_bw
            self.logits = tf.contrib.layers.fully_connected(out, 2)

    def add_loss(self):
        labels = self.labels
        gits = self.logits
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gits,
                                                              labels=labels)
        self.loss = tf.reduce_mean(loss, name='Loss')
        tf.summary.scalar('loss', self.loss)

    def run_train_iter(self, sess, ids, mask, labels, summary_writer):
        """
        This performs a single training iteration
        (forward pass, loss computation, backprop, parameter update)
        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard
        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.rule_ids] = ids
        input_feed[self.rule_mask] = mask
        input_feed[self.labels] = labels
        input_feed[self.keep_prob] = 1.0

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates,
                       self.summaries,
                       self.loss,
                       self.global_step,
                       self.param_norm,
                       self.gradient_norm]

        # Run the model
        out = sess.run(output_feed, feed_dict=input_feed)
        [_, summaries, loss, global_step, param_norm, gradient_norm] = out

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def train(self, descriptions, labels):
        # make descriptions into 1 hot id vectors
        sess = self.sess
        ids, mask = self.get_ids_and_mask(descriptions)
        epoch = 0
        loss = 1
        summary_writer = tf.summary.FileWriter('./', sess.graph)
        self.sess.run(tf.global_variables_initializer())
        while epoch < self.num_epochs and loss > 1e-4:
            epoch += 1
            trn = self.run_train_iter(sess, ids, mask, labels, summary_writer)
            loss, global_step, param_norm, grad_norm = trn
            if global_step < 5 or global_step % 100 == 0:
                print('epoch: {}\nloss: {}\ngrad norm: {}\nparam_norm: \
                      {}'.format(epoch, loss, grad_norm, param_norm))

    def get_predictions(self, session, rule_ids, rule_mask):
        input_feed = {}
        input_feed[self.rule_ids] = rule_ids
        input_feed[self.rule_mask] = rule_mask

        output_feed = [self.logits]
        dist = session.run(output_feed, input_feed)
        pred = np.argmax(dist[0], axis=1)
        return pred

    def get_ids_and_mask(self, descriptions):
        ids = [sentence_to_token_ids(desc, self.word2id)[1] for desc in descriptions]  # noqa
        ids = padded(ids, self.rule_length)
        ids = np.array(ids)
        mask = (ids != PAD_ID).astype(np.int32)
        return ids, mask

    def submit_guesses(self, rg):
        """
        rg: a guess for the rule
        determines whether the rule is correct or not
        If the rule is correct, ends the game.
        Otherwise, displays the prior guesses
        and the penalty
        """
        ids, mask = self.get_ids_and_mask(rg)
        true_rule = self.get_predictions(self.sess, ids, mask)
        print('true rule: {}'.format(true_rule))
        if true_rule[0] == 1:
            return "You win!"
        else:
            guess_cols = ["Rule Guess", "Penalty"]
            this_guess = pd.DataFrame(columns=guess_cols, data=[[rg, -1]])
            self.guesses = self.guesses.append(this_guess)
            return self.guesses

    def submit_list(self, seq):
        """
        l: array containing the list
        determines whether the list follows the sequence or not
        """
        assert(len(seq) == 3)

        def decider(x):
            return (x[0] < x[1] and x[1] < x[2])
        follows_rule = decider(seq)
        list_cols = ["Sequence", "Follows Rule"]
        new_row = pd.DataFrame(columns=list_cols,
                               data=[[str(seq), follows_rule]])
        self.lists = self.lists.append(new_row)
        return self.lists
