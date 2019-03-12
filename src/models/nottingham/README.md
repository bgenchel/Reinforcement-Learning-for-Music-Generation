# GUIDE TO THE FILES
## main.py
Runs the training procedure and saves trained models. Run python main.py -h to see command line args.

## generator.py
Holds and defines the generator class, a 1-layer LSTM-RNN

## discriminator.py
Holds and defines the discriminator class, a CNN with a 2 entry log softmax output

## rollout.py
Defines the rollout class, which calculates rewards for the generator's state based on the discriminator's judgements of it's outputs.
Implements the Monte Carlo Rollout

## data_iter.py
Defines dataset classes for iterating over the real.data and generated.data files used for training the discriminator

## eval.py
Runs evaluations for the model, including BLEU score and MGEval

## gan_loss.py
Defines how loss is calculated for the generator during adversarial training, incorporates rewards.
