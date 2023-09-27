from __future__ import unicode_literals, print_function, division
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터 준비
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Lang:
    def __init__(self):
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 2


# 데이터 정규화
def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace("[^A-z\s]+", " ")
    sentence = sentence.str.normalize("NFD")
    sentence = sentence.str.encode("ascii", errors="ignore").str.decode("utf-8")
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1)
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2):
    # 책에서 안알려준 other,
    # delimiter==sperator 는 어떤 문자를 기준으로 나눌지인데
    # 해당 파일에는 Creative commans 같은게 붙어있어서 그쪽에서 오류남
    df = pd.read_csv(loc, delimiter="\t", header=None, names=[lang1, lang2, "other"])
    return df


def process_data(lang1, lang2):
    df = read_file("./data2/%s-%s.txt" % (lang1, lang2), lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang = Lang()
    output_lang = Lang()
    pairs = list()
    for i in range(len(df)):
        if (len(sentence1[i].split(" ")) < MAX_LENGTH) and (
            len(sentence2[i].split(" ")) < MAX_LENGTH
        ):
            full = [sentence1[i], sentence2[i]]
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)

    return input_lang, output_lang, pairs


# 텐서로 변환
def indexesFromSentence(lang: Lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang: Lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# 인코더 네트워크
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embbed_dim = embbed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# 디코더 네트워크
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embbed_dim = embbed_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


# seq2seq 네트워크
class Seq2Seq(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, device, MAX_LENGTH=MAX_LENGTH
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
        decoder_hidden = encoder_hidden.to(device)
        decoder_input = torch.tensor([SOS_token], device=device)

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = output_lang[t] if teacher_force else topi
            if not teacher_force and input.item() == EOS_token:
                break
        return outputs


# 모델의 오차 계산 함수 정의
teacher_forcing_ratio = 0.5


def Model(model, input_tensor, target_tensor, model_optim, criterion):
    model_optim.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])
    loss.backward()
    model_optim.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


# 모델 훈련 함수 정의
def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [
        tensorsFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(num_iteration)
    ]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            avg_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print("%d %.4f" % (iter, avg_loss))

    torch.save(model.state_dict(), "./data2/mytraining.pt")
    return model


# 모델 평가
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])
        decoded_words = list()
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)

            if topi[0].item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("input {}".format(pair[0]))
        print("output {}".format(pair[1]))
        output_words = evaluate(model, input_lang, output_lang, pair)
        output_sentences = " ".join(output_words)
        print("predicted {}".format(output_sentences))


# 모델 평가
lang1 = "eng"
lang2 = "fra"
input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print("random sentence {}".format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print("Input: {} Output: {}".format(input_size, output_size))

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 75000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

print(encoder)
print(decoder)

model = trainModel(model, input_lang, output_lang, pairs, num_iteration)


# 임의의 문장에 대한 평가 결과
evaluateRandomly(model, input_lang, output_lang, pairs)


# 어텐션이 적용된 디코더
class AttenDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weight = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1,
        )
        attn_applied = torch.bmm(
            attn_weight.unsqueeze(0),
            encoder_outputs.unsqueeze(0),
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        outpu = self.out(output[0])
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weight


# 어텐션 디코더 모델 학습을 위한 함수
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, lr=0.01):
    start = time.time()
    plot_losses = list()
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)
    training_pairs = [
        tensorsFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % 5000 == 0:
            print_loss_avg = print_loss_total / 5000
            print_loss_total = 0
            print("%d, %.4f" % (iter, print_loss_avg))


# 어텐션 디코더 모델 훈련
embed_size = 256
hidden_size = 512
num_layers = 1
input_size = input_lang.n_words
output_size = output_lang.n_words

encoder1 = Encoder(input_size, hidden_size, embed_size, num_layers)
attn_decoder1 = AttenDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)

print(encoder1)
print(attn_decoder1)

attn_model = trainIters(
    encoder1,
    attn_decoder1,
    75000,
    print_every=5000,
    plot_every=100,
    lr=0.01,
)
