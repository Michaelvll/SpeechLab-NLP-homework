import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import utils.common as common


class BiLstm_CRF(nn.Module):
    def __init__(self, word2idx, tag2idx, embedding_dim, hidden_dim, num_layers=1, device=None):
        super(BiLstm_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.tag2idx = tag2idx
        self.word2idx = word2idx
        self.word_size = len(word2idx)
        self.tag_size = len(tag2idx)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.word_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim //
                            2, num_layers=num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)
        self.crf = nn.Parameter(torch.randn(
            self.tag_size, self.tag_size).to(device))

        self.crf.data[:, self.tag2idx['<START>']] = -10000
        self.crf.data[self.tag2idx['<END>'], :] = -10000

        self.hidden = self.init_hidden()

        self.device = device

    def init_hidden(self):
        return (torch.randn(2 * self.num_layers, 1, self.hidden_dim // 2), torch.randn(2 * self.num_layers, 1, self.hidden_dim // 2))

    def _prepare_sentence(self, sentence):
        return torch.tensor([self.word2idx[x] if x in self.word2idx else self.word2idx['<unk>'] for x in sentence])

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def _crf_score_total(self, features):
        score_previous = torch.full((1, self.tag_size), -10000.0)
        score_previous[0, self.tag2idx['<START>']] = 0.0
        for feature in features:
            score_previous = torch.t(score_previous).expand(-1, self.tag_size)
            feature_expand = feature.expand(self.tag_size, -1)
            score = score_previous + feature_expand + self.crf
            score_previous = common.log_sum_exp(score).view(1, -1)
        score_end = score_previous + \
            self.crf[:, self.tag2idx['<END>']].view(1, -1)
        return common.log_sum_exp(torch.t(score_end))

    def _crf_score_best(self, features, tags):
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag2idx['<START>']], dtype=torch.long), tags])
        for i, feature in enumerate(features):
            score += self.crf[tags[i], tags[i+1]] + feature[tags[i + 1]]
        score += self.crf[tags[-1], self.tag2idx['<END>']]
        return score

    def _viterbi_labeling(self, features):
        score_previous = torch.full((1, self.tag_size), -10000.0)
        score_previous[0, self.tag2idx['<START>']] = 0.0

        steps = []
        scores = []
        for feature in features:
            score_previous = torch.t(score_previous).expand(-1, self.tag_size)
            feature_expand = feature.expand(self.tag_size, -1)
            score = score_previous + feature_expand + self.crf
            step = torch.argmax(score, dim=0)
            steps.append(step)
            score_previous = score[step, torch.arange(
                score.size(0))].view(1, self.tag_size)
            scores.append(score_previous)
        score_end = score_previous + \
            self.crf[:, self.tag2idx['<END>']].view(1, -1)
        scores.append(score_end)

        best_end = torch.argmax(score_end).item()
        score_path = score_end[0, best_end]

        labels = [best_end]
        for step in reversed(steps):
            labels.append(step[labels[-1]].item())
        start = labels.pop()
        assert start == self.tag2idx['<START>']
        labels.reverse()
        return labels, score_path

    def loss(self, sentence, tags):
        sentence = self._prepare_sentence(sentence)
        lstm_feature = self._get_lstm_features(sentence)
        score_total = self._crf_score_total(lstm_feature)
        score_best = self._crf_score_best(lstm_feature, tags)
        return score_total - score_best

    def forward(self, sentence):
        sentence = self._prepare_sentence(sentence)
        lstm_feature = self._get_lstm_features(sentence)
        labels, score = self._viterbi_labeling(lstm_feature)
        return labels, score
