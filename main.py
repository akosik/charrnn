import argparse
import string
from random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CharRnn
from utils.data_prep import TinyShakespeareCharRnnDataset
from train import train, test
from generate import generate

def main():
    parser = argparse.ArgumentParser(description="CharRnn")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--rnn-type', type=str, default="LSTM",
                        help='RNN Cell Type to use')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout prob (default: 0.5)')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='number of hidden units (default: 300)')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    num_tokens = len(string.printable)
    model = CharRnn(args.rnn_type, 1, args.hidden_size,
                    num_tokens, args.layers, dropout=args.dropout).to(device)

    train_loader = torch.utils.data.DataLoader(
        TinyShakespeareCharRnnDataset('data/tinyshakespeare.txt', train=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        TinyShakespeareCharRnnDataset('data/tinyshakespeare.txt', train=False),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion)
        primer = choice(string.printable)
        print(generate(args, model, device, prime_string=primer))

    if (args.save_model):
        torch.save(model.state_dict(),"tshake_crnn.pt")


if __name__ == "__main__":
    main()
