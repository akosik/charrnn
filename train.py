import torch
import torch.nn.functional as F
import torch.optim as optim

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        rnn_hidden = model.init_hidden(data.size(0))
        output, _ = model(data, rnn_hidden)
        loss = criterion(output.permute(0, 2, 1), target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("Train epoch: {} [{}/{} {:.0f}%]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            rnn_hidden = model.init_hidden(data.size(0))
            output, _ = model(data, rnn_hidden)
            test_loss += criterion(output.permute(0,2,1), target).item()
            pred = output.argmax(dim=2, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            incorrect += pred.ne(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)*target.size(1)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, correct+incorrect,
        100. * correct / (correct+incorrect)))
