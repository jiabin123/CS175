import string
import random
import os
import time
import math
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_words = ""
words = open('imdb.vocab', encoding='utf-8').read().strip().split('\n')
for w in words:
    all_words += w;
n_words = len(all_words)

category_lines = {}   # a dict mapping each category to a list of lines
all_categories = [0, 1]
n_categories = len(all_categories)

# generate data for category_lines & all_categories
path = './data'
pos_reviews = []
neg_reviews = []
for f_name in os.listdir('./data'):
	file = open((path + f_name), 'r')
	rating = int(f_name[-5])
	review = file.read().lower()
	if rating < 5:
		neg_reviews.append(review)
	if rating == 0:
		pos_reviews.append(review)
	else:
		pos_reviews.append(review)
	file.close()

category_lines[0] = neg_reviews
category_lines[1] = pos_reviews

# Turn words into tensors
# Find word index from all_letters, e.g. "the" = 0
def wordToIndex(word):
    return all_words.find(word)


# Just for demonstration, turn a word into a <1 x n_words> Tensor
def wordToTensor(word):
    tensor = torch.zeros(1, n_words)
    tensor[0][wordToIndex(word)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_words> tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_words)
    for li, word in enumerate(line):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        # initialize the hidden_size
        self.hidden_size = hidden_size

        # 1st layer: combine both input + hidden, output to hidden layer
        # 2nd layer: combine both input + hidden, output to output
        # use LogSoftmax for output
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # combine input and hidden together
        # feed into hidden state
        # finally, feed into output state. Use softmax to output
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_words, n_hidden, n_categories)

def train_iteration_CharRNN(learning_rate, category_tensor, line_tensor):
    criterion = nn.NLLLoss()
    hidden = rnn.initHidden()
    rnn.zero_grad()

    # The forward process
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # The backward process
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate) # use NEGATIVE value cuz gradient is reduced

    return output, loss.item()


def train_charRNN(n_iters, learning_rate):
    print_every = 1000
    current_loss = 0

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train_iteration_CharRNN(learning_rate, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            print('Average loss: %.4f' % (current_loss/print_every))
            current_loss = 0

    torch.save(rnn, 'char-rnn-classification.pt')


def predict(input_line, n_predictions=8):
    print("Prediction for %s:" % input_line)
    hidden = rnn.initHidden()

    # Generate the input for RNN
    line_tensor = lineToTensor(input_line)
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # Get the value and index of top K predictions from the output
    # Then apply Softmax function on the scores of all category predictions so we can
    # output the probabilities that this name belongs to different languages.
    topv, topi = output.topk(n_predictions, 1, True)
    softmax = torch.softmax(output, 1);
    top_prob, probi = softmax.topk(n_predictions, 1, True)  # do NOT forget the 2nd output! (probi)
    predictions = []  # store the final prediction

    for i in range(n_predictions):
        value = topv[0][i].item()
        prob = top_prob[0][i].item() * 100  # convert it to a percentage number
        category_index = topi[0][i].item()
        print('%s Probability: (%.2f), Score: (%.2f)' % (all_categories[category_index], prob, value))
        predictions.append([value, all_categories[category_index]])
    return predictions


# RUN
print("start running")
train_charRNN(5000, 0.05)
#predict("Hanzawa")
