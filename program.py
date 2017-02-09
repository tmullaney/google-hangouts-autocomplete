'''
program.py

Given a trained model and a context, outputs either autocomplete options or sampled text.

Arguments:
    mode: autocomplete OR sample
    model_dir: directory containing model and character map produced by train.py

Example call:
    python program.py autocomplete ./2017-02-07 00:37:16.540126

Example context: 
<Other Person> yo
<Other Person> have you seen this?
<Tommy Mullaney> 

Example autocomplete:
1. hahaha
2. yeah that's awesome
3. yeah that's pretty cool
4. yeah that's pretty much
5. yeah that's pretty sure

Example sample:
<Other Person> although the recopding is going to be a little series about the presentation is going to be a but we can do it
<Other Person> and I can't wait until here
<Other Person> hahaha
<Tommy Mullaney> what about the cool schemically sounds good to me
<Other Person> how's the best enough haha
<Other Person> alright I'm going to try the over a little bit of a second readons
<Other Person> hahaha yeah but you can see a nice but of stuff like that are more super

'''

import numpy as np
import tensorflow as tf
import sys
import os

from model import CharRNN

# Read command line args
mode = sys.argv[1]
model_dir = sys.argv[2]

# Use most recent checkpoint
checkpoint_filename = [x for x in os.listdir(model_dir) if x[-4:] == 'ckpt'][-1]

# Load vocabulary
print('Loading vocabulary...')
with open(os.path.join('.', model_dir, 'char-map.txt')) as f:
    vocab = [chr(int(line)) for line in f.readlines()]

# Restore model
print('Loading model...')
model = CharRNN(vocab)

# Start session and poll for text to autocomplete
with tf.Session() as sess:
    model.restore_checkpoint(sess, os.path.join('.', model_dir, checkpoint_filename))
    input_seq = None
    while True:
        if mode == 'autocomplete':
            print('---')
            print('Enter context (or type \'quit\' to quit):')
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    break

            if lines[0] == 'quit':
                print('Goodbye...')
                break # exit program

            print('Thinking...')
            input_seq = '\n'.join(lines)
            results = model.autocomplete(sess, input_seq, n=5)
            print('Autocomplete:')
            for i, r in enumerate(results):
                print('{0}. {1}'.format(i+1, r[len(input_seq):][:-1])) # skip context and trailing newline char
        else:
            print('---')
            print('Hit enter to sample more characters (or type \'quit\' to quit):')
            line = input()
            if line == 'quit':
                print('Goodbye...')
                break # exit program
            
            print('Sampling...')
            if input_seq is None:
                input_seq = np.random.choice(vocab) # pick random character
            sample = model.sample(sess, input_seq=input_seq, num_to_sample=500, temperature=0.5)
            input_seq += sample
            print(sample)

