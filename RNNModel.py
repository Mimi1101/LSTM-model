import torch
from torch import nn
import numpy as np
import string

#so this is a RNN Model class with the given parameters in the assignment
class RNNModel(nn.Module):
    def __init__(self, max_len):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(28, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 28)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

def create_model(max_len):
    model = RNNModel(max_len)
    return model

def train_model(model, n_epochs):
    #reading the file
    with open("yob2018.txt", "r") as f:
        data = f.readlines()
    names = []
    for line in data:
        #converting the letters to lower case
        name = line.strip().split(",")[0].lower()
        if all(c in string.ascii_lowercase for c in name):
            names.append(name)
    max_len = max(len(name) for name in names) + 2  # Adding plus 2 for the beginning and end markers

    char_to_idx = {char: i for i, char in enumerate(string.ascii_lowercase + " ")}
    #input sequences
    data_input = torch.zeros(len(names), max_len, 28)
    #output sequences
    data_output = torch.zeros(len(names), max_len, dtype=torch.long)

 #to process the names and populate the data_input and data_output tensors. 
    for i, name in enumerate(names):
        name_with_markers = " " + name + " "  # Adding beginning and end markers
        if len(name_with_markers) < max_len:
            name_with_markers += " " * (max_len - len(name_with_markers))  # Padding with spaces

        for j in range(max_len):
            data_input[i, j, char_to_idx[name_with_markers[j]]] = 1
            if j > 0:
                data_output[i, j - 1] = char_to_idx[name_with_markers[j]]
        data_output[i, max_len - 1] = char_to_idx[" "] 

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignoring the padding positions as specified in the assignment
    optimizer = torch.optim.Adam(model.parameters())

 #training the model for 20 epochs
    for epoch in range(n_epochs):
        loss_sum = 0
        for i in range(len(data_input)):
            optimizer.zero_grad()
            output = model(data_input[i].unsqueeze(0))
            target = data_output[i]
            valid_indices = target != char_to_idx[" "]
            loss = criterion(output.view(-1, 28)[valid_indices], target[valid_indices])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_sum/len(data_input)}")
    torch.save(model.state_dict(), "finalmodel.pth")


def load_trained_model(max_len):
    model = create_model(max_len)
    model.load_state_dict(torch.load("finalmodel.pth"))
    return model

def generate_name(M, N):
    M.eval()
    char_to_idx = {char: i for i, char in enumerate(string.ascii_lowercase + " ")}
    idx_to_char = {i: char for i, char in enumerate(string.ascii_lowercase + " ")}

    input_seq = torch.zeros(1, 1, 28)
    input_seq[0, 0, char_to_idx[" "]] = 1

    generated_name = ""
    for i in range(N):
        output = M(input_seq)
        prob = torch.softmax(output[0, -1], dim=0)
        next_char_idx = torch.multinomial(prob, num_samples=1).item()

        if next_char_idx == char_to_idx[" "]:  # Beginning-of-name character
            continue
        if next_char_idx == len(idx_to_char) - 1:  # End-of-name character
            break

        next_char = idx_to_char[next_char_idx]
        generated_name += next_char
        # Shifting the input sequence
        input_seq = torch.zeros(1, 1, 28)
        input_seq[0, 0, next_char_idx] = 1

    return generated_name


if __name__ == "__main__":
    mlen = 15 
    um = create_model(mlen)
    train_model(um, 20)
    tm = load_trained_model(mlen)

    for i in range(5):
        print(generate_name(tm, 5))
