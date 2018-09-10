import torch
import torch.nn as nn
import torch.nn.functional as F


class Solver():
    def __init__(self, model, device,
                 params=dict(), epoch=400, batch_size=400, print_every=10, save_best=False):
        self.model = model
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.print_every = print_every

        params.setdefault('lr', 1e-3)
        params.setdefault('weight_decay', 0.01)
        self.params = params
        self.save_best = save_best
        print(params['lr'])

    def train(self, x, y, loss_function, dev_x=None, dev_y=None):
        batch = Batch(len(y), batch_size=1)
        print_dev = False
        print_count = 0
        if dev_x is not None and dev_y is not None:
            # best_dev_acc = 0.96
            print_dev = True
            dev_y = torch.tensor(dev_y).to(self.device)
            dev_size = dev_y.size()[0]

        self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        while batch.epoch < self.epoch:
            self.model.zero_grad()
            batch_x, batch_y = batch.get_batch((x, y))
            # print(batch_x.type())
            loss = loss_function(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            batch.next()

            # if print_dev and print_count % self.print_every == 0:
            #     with torch.no_grad():
            #         self.predict(dev_x, dev_y)

            #         train_acc = float(
            #             torch.sum(train_label == torch.tensor(batch_y).to(self.device))) / self.batch_size
            #         # print(dev_label)
            #         print(("[iter {}, epoch {}]> Training Loss: {},  Training batch accuracy: {}\n> Dev Loss: {}, Dev accuracy:{}").format(
            #             print_count, batch.epoch, loss.item(), train_acc, dev_loss.item(), dev_acc))
            #         self.model.train()
            # print_count += 1
        self.save_model()

    def predict(self, x_test, y_test=None, filename='predict.txt'):
        self.model.eval()
        idxs = list(range(len(x_test)))
        if self.need_sort:
            x_test = Data.process_x2tensor(x_test)
            x_test, idxs = sort_data((x_test, idxs), reverse=True)
        # print(x_test)
        out = self.model(x_test)
        label = torch.argmax(out, dim=1).cpu()
        label = label.numpy().tolist()
        # print(list(zip(idxs, label)))
        if self.need_sort:
            idxs, label = resort(idxs, label)

        if y_test is not None:
            score = torch.sum(torch.tensor(label) == torch.tensor(y_test))
            print("test accuracy:", float(score) / len(y_test))

        with open(filename, 'w', newline='\n') as outfile:
            print('id,classes', file=outfile)
            for i in range(len(idxs)):
                print('{},{}'.format(idxs[i] + 1, label[i] + 1), file=outfile)

    def save_model(self, filename='0001.model'):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename='0001.model'):
        self.model.load_state_dict(torch.load(filename))
