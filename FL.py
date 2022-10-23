#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code was utilized and modified from the version available at
# https://github.com/yonetaniryo/federated_learning_pytorch/blob/master/FL_pytorch.ipynb
# with the following license.

# MIT License

# Copyright (c) 2021 Ryo Yonetani

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20480, 2048)
        self.fc2 = nn.Linear(2048, 10)
        self.dropout = nn.Dropout(0.10)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

class FedMLFunc:

    def client_update(self, client_model, optimizer, train_loader, epoch=5):
        correct = 0
        total = 0
        for e in range(epoch):
            for (data, target) in train_loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return [loss.item(),correct/total]

    def server_aggregate(self, global_model, client_models):
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    def test(self, global_model, test_loader, test_dataset):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataset.dataset)
        acc = correct / len(test_dataset.dataset)
        return test_loss, acc
