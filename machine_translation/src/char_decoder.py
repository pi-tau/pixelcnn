"""Character Decoder module for the Neural Machine Translation (NMT) model.
The character-level decoder is used to replace unknown words with words generated one
character at a time. This produces rare and out-of-vocabulary target words.
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module): 
    """Character-level language model for the target language.

    Attributes:
        target_vocab (VocabEntry): Vocabulary for the target language.
        char_emb (nn.Embedding): Embedding layer transforming single characters into
            dense vectors.
        charDecoder (nn.LSTM): LSTM recurrent neural net acting as a language model.
        char_output_projection (nn.Linear): Linear layer producing scores over the
            charecters in the vocabulary.
    """

    def __init__(self, hidden_size, char_embed_size, target_vocab):
        """Init Character Decoder.

        Args:
            hidden_size (int): Hidden size of the decoder LSTM.
            char_embed_size (int): Dimensionality of character embeddings.
            target_vocab (VocabEntry): Vocabulary for the target language.
                See vocab.py for documentation.
        """
        super().__init__()
        self.target_vocab = target_vocab

        self.char_emb = nn.Embedding(len(target_vocab.char2id), char_embed_size,
                                           padding_idx=target_vocab.char_pad)
        self.charDecoder = nn.LSTM(char_embed_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))

    def forward(self, input, dec_hidden):
        """Forward pass of character decoder.

        Args:
            input (Tensor): Tensor of integers, shape (max_word_len, sent_len * batch_size).
            dec_hidden (tuple(Tensor, Tensor)): Internal state of the LSTM before reading
                the input characters. A tuple of two tensors of shape (1, batch, hidden_size).

        Returns:
            scores (Tensor): Tensor of shape (max_word_len, sent_len * batch_size, vocab_size).
            dec_hidden (tuple(Tensor, Tensor)): Internal state of the LSTM after reading
                the input characters. A tuple of two tensors of shape (1, batch, hidden_size).
        """
        x = self.char_emb(input)    # (word_len, sent_len * batch_size, embed_size)
        output, (h_n, c_n) = self.charDecoder(x, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, (h_n, c_n)

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding.

        Args:
            initialStates (tuple(Tensor, Tensor)): Initial internal state of the LSTM.
                A tuple of two tensors of size (1, batch, hidden_size).
            device (torch.device): Indicates whether the model is on CPU or GPU.
            max_length (int, optional): Maximum length of sequence of chars to decode.
                Defaults to 21.
        
        Returns:
            decodedWords (List[str]): A list (of length batch_size) of strings, each of
                which has length <= max_length. The decoded strings do NOT contain the
                start-of-word and end-of-word characters.
        """
        _, batch_size, _ = initialStates[0].shape

        start = self.target_vocab.start_of_word
        current_chars = [[start] for _ in range(batch_size)]
        current_chars = torch.tensor(current_chars, device=device).t()
        output_words = []
        dec_hidden = initialStates
        for i in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)
            _, current_chars = torch.max(scores, dim=-1)
            output_words.append(current_chars)

        output_words = torch.cat(output_words, dim=0).t().tolist()
        decodedWords = []
        for word in output_words:
            current_word = ""
            for idx in word:
                if idx == self.target_vocab.end_of_word:
                    break
                else:
                    char = self.target_vocab.id2char[idx]
                    current_word += char
            decodedWords.append(current_word)

        return decodedWords

#