"""Embeddings for the Neural Machine Translation (NMT) model.
Consists of word embeddings for one language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEmbeddings(nn.Module):
    """Class that converts input words to their embeddings.
    The class uses a character-based CNN to construct word embeddings.

    Attributes:
        word_embed_size (int): Embedding size for the output word.
        char_embed_size (int): Embedding size for the characters.
        embed (nn.Embedding): Embedding layer transforming single characters into
            dense vectors.
        conv (nn.Conv1d): Convolutional layer for character-based CNN.
        gate (nn.Linear): Linear layer for highway network.
        proj (nn.Linear): Linear layer for highway network.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, word_embed_size, char_embed_size, vocabentry, kernel_size=5,
                 padding=1, dropout_rate=0.3):
        """Init the Embedding layer for one language.

        Args:
            word_embed_size (int): Embedding size for the output word.
            char_embed_size (int): Embedding size for the characters. See vocab.py for
                documentation.
            kernel_size (int, optional): Kernel size for the character-level CNN.
                Defaults to 5.
            padding (int, optional): Size of padding for the character-level CNN.
                Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.embed = nn.Embedding(len(vocabentry.char2id), char_embed_size)
        self.conv = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size,
                              kernel_size=kernel_size, padding=padding)
        self.gate = nn.Linear(word_embed_size, word_embed_size)
        self.proj = nn.Linear(word_embed_size, word_embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_padded):
        """Look up character-based CNN embeddings for the words in a batch of sentences.

        Args:
            x_padded (Tensor): Tensor of shape (sent_length, batch_size, max_word_length)
                of integers where each integer is an index into the character vocabulary. 

        Returns:
            x_wordEmb (Tensor): Tensor of shape (sent_length, batch_size, word_embed_size),
                containing the CNN-based embeddings for each word of the sentences in the
                batch.
        """
        sent_len, batch_size, max_word_length = x_padded.shape

        # For each character look-up a dense character vector.
        x_emb = self.embed(x_padded)

        # Reshape "x_emb". PyTorch Conv1d performs convolution only on the last dimension of the input.
        x_reshaped = x_emb.permute(0, 1, 3, 2)
        x_reshaped = x_reshaped.reshape(sent_len * batch_size, self.char_embed_size, max_word_length)

        # Combine the character embeddings using a convolutional layer. L_out = L_in + 2*padding - kernel_size + 1
        x_conv = F.relu(self.conv(x_reshaped))          # (sent_len * batch_size, word_embed_size, L_out)
        x_conv, _ = torch.max(x_conv, dim=-1)           # max pool

        # Use a highway network.
        x_gate = torch.sigmoid(self.gate(x_conv))
        x_proj = F.relu(self.proj(x_conv))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv

        # Apply dropout.
        x_wordEmb = self.dropout(x_highway)
        x_wordEmb = x_wordEmb.reshape(sent_len, batch_size, self.word_embed_size)

        return x_wordEmb

#