'''
train.py

Trains a CharRNN model on a supplied input text file formatted like this:
    <Tommy Mullaney>hey
    <Other Person>what's up?
    <Tommy Mullaney>you around on sat?
    [...]

Outputs:
    - checkpoint files every 10000 iterations
    - loss_history.csv
    - acc_history.csv

Example:
    python train.py hangouts-chat-log.txt

'''

import tensorflow as tf
import numpy as np
import datetime
import sys
import os

from model import CharRNN

TRAINING_ITERS = 100000 # 1000 takes about 2 hrs
BATCH_SIZE = 100 # sequences

### Load and preprocess data
# Create directory to store output
run_name = str(datetime.datetime.now())
os.makedirs(run_name)

# Load input data
print('Reading input: {0}'.format(sys.argv[1]))
input_file = open(sys.argv[1]).read()

# Convert characters to one-hot vectors
chars = sorted(list(set(input_file)))
ord_strs = [str(ord(c)) for c in chars]
input_one_hot = np.zeros([len(input_file), len(chars)])
for i in range(len(input_file)):
    c = input_file[i]
    input_one_hot[i, chars.index(c)] = 1
print('Dataset shape:', input_one_hot.shape)
with open(os.path.join('.', run_name, 'char-map.txt'), mode='w') as f:
    f.write('\n'.join(ord_strs))

# Create CharRNN model
model = CharRNN(chars)

### Train for a while
with tf.Session() as sess:
    model.initialize(sess)

    pos = 0
    epoch = 1    
    n_steps = model.n_steps
    for i in range(TRAINING_ITERS):
        # Generate next batch of sequences
        batch_x = []
        batch_y = []
        batch_seqlen = []
        for _ in range(BATCH_SIZE):
            if pos+n_steps+1 >= len(input_one_hot):
                pos = np.random.choice(range(n_steps)) # go back to random starting point
                epoch += 1
            seq_x = input_one_hot[pos:pos+n_steps]
            seq_y = input_one_hot[pos+1:pos+n_steps+1]
            batch_x.append(seq_x)
            batch_y.append(seq_y)
            batch_seqlen.append(len(seq_x))
            pos += n_steps
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_seqlen = np.array(batch_seqlen)
        
        # Train model on batch
        model.train_step(sess, batch_x, batch_y, batch_seqlen)
        
        if i % 100 == 0:
            # Write loss and accuracy to log
            batch_loss, batch_acc = model.calc_loss_acc(sess, batch_x, batch_y, batch_seqlen)
            with open(os.path.join('.', run_name, 'loss-history.txt'), mode='a') as f:
                f.write('{0},{1}\n'.format(i, batch_loss))
            with open(os.path.join('.', run_name, 'acc-history.txt'), mode='a') as f:
                f.write('{0},{1}\n'.format(i, batch_acc))

            # Print status
            if i % 1000 == 0:
                print(datetime.datetime.now(), 
                      '| iter', i, 
                      'batch_loss:', batch_loss, 
                      'batch_acc:', batch_acc)
                print(repr(model.sample(sess, input_seq=np.random.choice(chars), num_to_sample=200)))

                # Save checkpoint
                if i % 10000 == 0:
                    save_path = model.save_checkpoint(sess, os.path.join('.', run_name, 'checkpoint-{0}.ckpt'.format(i)))
                    print('Checkpoint saved:', save_path)
                
                print('--')
            
