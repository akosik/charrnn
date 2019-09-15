import unidecode
import string
import torch
from torch.utils.data import Dataset

class TinyShakespeareCharRnnDataset(Dataset):
    def __init__(self, text_file, chunk_len=200, transform=None, train=True):

        self.chunk_len = chunk_len
        self.transform = transform

        self.char2id = {c: i for i,c in enumerate(string.printable)}

        self.char_list = [ch for ch in unidecode.unidecode(open(text_file).read())]
        if train:
            self.char_list = self.char_list[:int(len(self.char_list)*.8)]
        else:
            self.char_list = self.char_list[int(len(self.char_list)*.8):]

    def __len__(self):
        return len(self.char_list) // self.chunk_len

    def __getitem__(self, idx):
        char_list_chunk = self.char_list[idx*self.chunk_len:(idx+1)*self.chunk_len]
        charid_tensor = torch.tensor([self.char2id[c] for c in char_list_chunk], dtype=torch.long).reshape(self.chunk_len, 1)
        input_charid_tensor, output_charid_tensor = charid_tensor[:-1], charid_tensor[1:].squeeze()
        sample = {'input': input_charid_tensor, 'output': output_charid_tensor}

        if self.transform:
            self.transform(sample)

        return input_charid_tensor, output_charid_tensor
