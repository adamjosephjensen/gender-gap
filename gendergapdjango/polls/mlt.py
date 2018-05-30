import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
import re
from .vocab import PAD_ID, UNK_ID, get_glove
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
        self.num_epochs = 100
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
                print('epoch: {}>>LOSS: {}\ngrad norm: {}param_norm: \
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

    def submit_rule(self, rg):
        """
        rg: a guess for the rule
        determines whether the rule is correct or not
        If the rule is correct, ends the game.
        Otherwise, displays the prior guesses
        and the penalty
        """
        ids, mask = self.get_ids_and_mask([rg])
        true_rule = self.get_predictions(self.sess, ids, mask)
        print('true rule: {}'.format(true_rule))
        if true_rule[0] == 1:
            return True
        else:
            return False

    def submit_list(self, seq):
        """
        l: array containing the list
        determines whether the list follows the sequence or not
        """
        assert(len(seq) == 3)

        def decider(x):
            return (x[0] < x[1] and x[1] < x[2])
        follows_rule = decider(seq)
        return follows_rule

data = [
    ["increasing integers",1],
    ["increasing nationalism",0],
    ["increasing physics",0],
    ["increasing fish",0],
    ["increasing numbers",1],
    ["a series of increasing numbers",1],
    ["three numbers that increase in their values",1],
    ["numbers that are increasing",1],
    ["the numbers are in increasing order",1],
    ["the digits are in increasing order",1],
    ["each number is bigger than the previous number",1],
    ["each number is strictly bigger than the prior number",1],
    ["the sequence increases",1],
    ["the sequence increases",1],
    ["they all go up",1],
    ["the next one is greater than the prior one",1],
    ["each number is smaller than the next number",1],
    ["each number is strictly less than the next number",1],
    ["each number is greater than the previous number",1],
    ["the numbers go up by some amount",1],
    ["the numbers go up",1],
    ["the numbers are in increasing order",1],
    ["the digits are in increasing order", 1],
    ["each number is bigger than the previous number",1],
    ["each number is strictly bigger than the prior number",1],
    ["the sequence increases",1],
    ["the sequence increases",1],
    ["they all go up", 1],
    ["the next one is greater than the prior one", 1],
    ["each number is smaller than the next number",1],
    ["each number is strictly less than the next number",1],
    ["each number is greater than the previous number",1],
    ["the numbers go up by some amount",1],
    ["the numbers go up",1],
    ["the list strictly increases",1],
    ["the next number needs to be bigger than the number before it",1],
    ["the list climbs",1],
    ["the sequence strictly increases",1],
    ["the sequence strictly goes up",1],
    ["the numbers are in increasing order",1],
    ["the numbers go up by two",0],
    ["each number is the next even number",0],
    ["the numbers are subsequent even numbers",0],
    ["the numbers are subsequent odd numbers",0],
    ["each number goes up by 1",0],
    ["I have six cats in my garage",0],
    ["number",0],
    ["evens",0],
    ["odds", 0],
    ["they increase",1],
    ["they go up",1],
    ["the rule is that",0],
    ["bigger than the one before it",1],
    ["add two each time",0],
    ["for anyone whose tried to text or call me in the past 2 weeks", 0],
    ["I got rid of that phone so I could focus on these albums", 0],
    ["rules are structure for people who can’t carve their own path", 0],
    ["free thinking is a super power", 0],
    ["I am hyper focused on the now", 0],
    ["take a walk outside fresh air is healing", 0],
    ["I can not wait for electric planes", 0],
    ["most fear is learned", 0],
    ["energy meeting. Beings from all different backgrounds", 0],
    ["in school we need to learn how magic built his business", 0],
    ["we need to have open discussions and ideas on unsettled pain", 0],
    ["We are all great artists", 0],
    ["we've invested in 3 companies since last week", 0],
    ["do meetings in different places and at different times", 0],
    ["I do not agree with this", 0],
    ["If someone tweeted that people who don’t drink are miserable", 0],
    ["This is a good point", 0],
    ["I will be doing none of the below", 0],
    ["why not debate your political positions", 0],
    ["positive masculinity is not toxic modern feminism is", 0],
    ["the tendency to search for", 0],
    ["when people falsely perceive an association", 0],
    ["they are weighing up the costs of being wrong", 0],
    ["rather than investigating in a neutral", 0],
    ["even scientists can be prone to confirmation bias", 0],
    ["maintain or strengthen beliefs in the face of contrary evidence", 0]
]

emb_matrix, word2id, id2word = get_glove('/Users/adamjensen/Documents/gender-gap/gendergapdjango/polls/glove.6B/glove.6B.50d.txt', 50) # make sure these match
model = GameModel(id2word, word2id, emb_matrix, 14)
desc, labels = zip(*data)
desc = list(desc)
labels = list(labels)
model.train(desc, labels)
