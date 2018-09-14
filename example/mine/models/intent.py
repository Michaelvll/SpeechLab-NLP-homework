import torch
import torch.nn as nn
import torch.nn.functional as F

class Intent(nn.Module):
    def __init__(self, hidden_size, target_size, device):
        super(Intent, self).__init__()
        print("build Intent...")
        d_a = 128
        self.W1 = nn.Linear(hidden_size, d_a)
        self.W2 = nn.Linear(hidden_size, d_a)
        self.attention = nn.Sequential(nn.Tanh(),
                                      nn.Linear(d_a, 1))
        #self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.hidden2label = nn.Sequential(nn.Linear(hidden_size, 128),
                                          nn.Dropout(),
                                          nn.ReLU(),
                                          nn.Linear(128, target_size))
        #self.hidden2label = nn.Linear(hidden_size, target_size)
        #self.loss = nn.NLLLoss(mask)
        #self.softmax = nn.AdaptiveLogSoftmaxWithLoss(512, target_size)

        self.softmax = nn.LogSoftmax(dim=1)
    def _forword_alg(self, feats):
        half_size = feats.size(2) // 2
        tail = torch.cat([feats[:,0, :half_size], feats[:,-1,half_size:]], dim=1)
        tail = self.W2(tail)
        tail = tail.view(tail.size(0), tail.size(1), 1).expand(tail.size(0), tail.size(1), feats.size(1)).transpose(1,2)
        fc_feats = self.W1(feats)
        a = self.softmax(self.attention(fc_feats+tail))
        m = self.tanh(torch.bmm(feats.transpose(1,2), a))
        m = m.view(feats.size(0), feats.size(2))

        output = self.softmax(self.hidden2label(m))
        return output

    def forward(self, feats):
        output = self._forword_alg(feats)
        return torch.argmax(output, dim=1), output

#    def neg_log_likelihood_loss(self, feats, labels):
#        output = self._forword_alg(feats)
#        return self.loss(output, labels)
#

