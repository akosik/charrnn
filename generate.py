import torch
import string

def generate(args, model, device, prime_string='A', predict_len=100, temperature=0.8):
    hidden = model.init_hidden(1)
    prime_tensor = torch.tensor([model.char2id[c] for c in prime_string], dtype=torch.long).reshape(-1, 1)
    predicted = prime_string

    # Use priming string to "build up" hidden state
    for p in range(len(prime_string) - 1):
        _, hidden = model(prime_tensor[p].reshape(1,1,1), hidden)
    inp = prime_tensor[-1]

    for p in range(predict_len):
        output, hidden = model(inp.reshape(1,1,1), hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = torch.tensor([model.char2id[c] for c in predicted_char], dtype=torch.long).reshape(-1, 1)
    return predicted
