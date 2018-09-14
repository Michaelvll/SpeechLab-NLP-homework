import utils.data_reader as data_reader


with open('data/train.txt', 'r') as infile:
    lines = [line.strip().split(' <=> ')[1] for line in infile.readlines()]

labels = {}
amt = 0
for line in lines:
    if line not in labels:
        labels[line] = 1
    else:
        labels[line] += 1
    amt += 1
for key, label in labels.items():
    labels[key] = label / amt
print(labels)
