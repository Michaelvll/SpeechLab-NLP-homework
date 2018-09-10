import utils.data_reader as data_reader

word2idx, idx2word = data_reader.read_vocab_from_data_file('./data/train.txt')
data_reader.save_vocab(idx2word, './mid/vocab.txt')
