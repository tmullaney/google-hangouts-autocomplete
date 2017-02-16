'''
model.py

Implements a character-level language model using a recurrent neural network (RNN).

'''

import tensorflow as tf
import numpy as np
from queue import PriorityQueue
import editdistance # https://pypi.python.org/pypi/editdistance

class CharRNN(object):
    def __init__(self, vocab):
        '''
        Initializes a CharRNN model for a given vocab_size.
        vocab: ordered list of characters in vocabulary
        '''
        # Vocab will be one-hot encoded
        self.vocab = vocab
        self.n_steps = 100 # max timesteps for RNN to remember a sequence
        self.lstm_size = 512 
        self.n_layers = 2
        self.learning_rate = 0.0001
        self.build_graph()
    
    def build_graph(self):
        '''
        Creates the TensorFlow graph. 
        Note: Resets the Tensorflow default graph.
        '''
        tf.reset_default_graph()

        # Input has shape [BATCH_SIZE, self.n_steps, vocab_size]
        # BATCH_SIZE can vary, so we represent it as 'None' in the placeholders
        # Note: shorter sequences must be padded with 0's so that they have length self.n_steps
        vocab_size = len(self.vocab)
        self.x = tf.placeholder(np.float32, shape=[None, self.n_steps, vocab_size])
        self.y = tf.placeholder(np.float32, shape=[None, self.n_steps, vocab_size])

        # Indicates each seq's real length (without zero-padding)
        # So TF won't compute the extra timesteps
        self.seqlen = tf.placeholder(tf.int32, [None])

        W_y = tf.Variable(tf.truncated_normal([self.lstm_size, vocab_size]))
        b_y = tf.Variable(tf.zeros([vocab_size]))

        # Reshape inputs
        # RNN API expects self.n_steps-length list of [BATCH_SIZE, vocab_size] tensors
        x_tr = tf.transpose(self.x, perm=[1, 0, 2]) # shape: [self.n_steps, BATCH_SIZE, vocab_size]
        x_re = tf.reshape(x_tr, [-1, vocab_size]) # shape: [self.n_steps*BATCH_SIZE, vocab_size]
        x_sp = tf.split(0, self.n_steps, x_re) # split into self.n_steps-length list of [BATCH_SIZE, vocab_size]

        # RNN returns:
        #  outputs: self.n_steps-length list of [BATCH_SIZE, vocab_size]
        #  state: final state
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.n_layers, state_is_tuple=True)
        outputs, state = tf.nn.rnn(stacked_lstm, x_sp, dtype=tf.float32, sequence_length=self.seqlen)

        # Reshape outputs to [self.n_steps*BATCH_SIZE, vocab_size]
        outputs = tf.reshape(tf.concat(1, outputs), [-1, self.lstm_size])

        # Output activation scores for each word in vocabulary
        self.logits = tf.matmul(outputs, W_y) + b_y

        # Reshape actual labels to same shape for accuracy comparison
        y_reshaped = tf.reshape(self.y, [-1, vocab_size])

        # Store loss, optimizer, and accuracy as ivars to call later
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, y_reshaped))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(self.logits,1), tf.argmax(y_reshaped,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def initialize(self, sess):
        '''
        Initializes graph in a given session.
        If you aren't restoring from a checkpoint, then you must call this before 
        doing any training or evaluating.
        '''
        sess.run(tf.initialize_all_variables())

    def restore_checkpoint(self, sess, checkpoint_file):
        '''
        Restores variable values from a checkpoint file.
        '''
        saver = tf.train.Saver()
        return saver.restore(sess, checkpoint_file)
    
    def save_checkpoint(self, sess, checkpoint_file):
        '''
        Saves a checkpoint to file.
        '''
        saver = tf.train.Saver()
        return saver.save(sess, checkpoint_file)

    def train_step(self, sess, batch_x, batch_y, batch_seqlen):
        '''
        Runs a training step for a single batch.
        batch_x and batch_y should be np.arrays 
        '''
        batch_feed = {self.x: batch_x, self.y: batch_y, self.seqlen: batch_seqlen}
        sess.run(self.optimizer, feed_dict=batch_feed)

    def calc_loss_acc(self, sess, batch_x, batch_y, batch_seqlen):
        '''
        Returns a tuple of the (loss, accuracy) for a single batch.
        '''
        batch_feed = {self.x: batch_x, self.y: batch_y, self.seqlen: batch_seqlen}
        return sess.run([self.loss, self.accuracy], feed_dict=batch_feed)

    def predict_next(self, sess, input_seq, temperature=1.0):
        '''
        Returns the predicted probability distribution for the next character,
        after a given input character sequence.
        '''
        # RNN can only look at most recent self.n_steps chars
        if len(input_seq) > self.n_steps:
            input_seq = input_seq[-self.n_steps:]

        # Convert input chars to zero-padded sequence of one-hot vectors
        seed_x = np.zeros([1, self.n_steps, len(self.vocab)]) # 1 batch
        for step, c in enumerate(input_seq):
            seed_x[0, step, self.vocab.index(c)] = 1

        # Predict scores and softmax probs
        scores = sess.run(self.logits, feed_dict={self.x:seed_x, self.seqlen:[len(input_seq)]})
        # probs = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature), axis=1, keepdims=True)
        norm = np.exp(scores - np.max(scores)) # normalize to prevent overflow
        probs = norm / np.array([np.sum(norm, axis=1)]).T
        
        return probs[len(input_seq)-1]
    
    def sample(self, sess, input_seq, num_to_sample, temperature=1.0):
        '''
        Sample a sequence of characters from the model, given an input character sequence.
        '''
        return_sequence = []
        for t in range(num_to_sample):
            probs = self.predict_next(sess, input_seq, temperature=temperature)

            # Sample next character from predicted probs
            next_char = np.random.choice(self.vocab, size=None, p=probs)
            input_seq += next_char
            return_sequence.append(next_char)
            
        return ''.join(return_sequence)

    def autocomplete(self, sess, input_seq, num_responses=5, n=20, m=10, verbose=False):
        '''
        Returns the `num_responses` most probable responses to a given input character sequence.
        Uses modified beam search terminated at a newline character.

        n: size of beam (the rest are pruned out)
        m: number of children to expand per node
        '''
        hypotheses = [] # each element is (-prob, sequence)
        hypotheses.append([-1.0, input_seq])
        results = []
        non_results = []

        depth = 0
        while len(results) < num_responses:
            depth += 1
            if verbose:
                print('--', depth)
            # Expand the top 'n' hypothesis's best 'n' next chars each
            hypotheses.sort()
            children = []
            for neg_p, seq in hypotheses[:n]:
                probs = self.predict_next(sess, seq)
                top_n_char_idx = list(np.argsort(probs)[::-1])[:m] # indices sorted from most to least probable
                for i in top_n_char_idx:
                    c = self.vocab[i]
                    p = probs[i]
                    new_seq = seq + c
                    if depth > 1 and p < 0.05:
                        continue # don't pursue low-probability options
                    if verbose and depth < 20:
                        print(p, '\t\t', repr(new_seq[len(input_seq):]), '\t\t', neg_p*p)
                    if c == '\n': 
                        if top_n_char_idx.index(i) == 0: # need high confidence in newline char
                            results.append(new_seq)
                            break # to increase diversity, stop pursuing this subtree
                    else:
                        children.append([neg_p*p, new_seq])
            
            # Children become the next round of hypotheses
            hypotheses = children

            if depth > 30:
                # Bail out, can't find a good autocomplete
                break

        return results[:num_responses]
