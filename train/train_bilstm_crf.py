import utils.solver as solver
from models.bilstm_crf import BiLstm_CRF
import utils.data_reader as data_reader


if __name__ == "__main__":
    (word2idx, idx2word), (tag2idx, idx2tag) = data_reader.read_word_and_tag(
        './mid/vocab.txt', './data/lab.txt')
    seqs_train, tags_train, intents_train = data_reader.read_seqtag_data_with_unali_act(
        './data/train.txt', word2idx, tag2idx)
    model = BiLstm_CRF(word2idx, tag2idx, EMBEDDING_DIM, HIDDEN_DIM, 1)

    pass
