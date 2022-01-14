"""Vocab object for the Neural Machine Translation (NMT) model.
Containts two separate vocabulary structures. One for the source language and one for the
target language.
Each vocabulary structure is a VocabEntry object.
"""

from collections import Counter
from itertools import chain
import json

import torch

from src.utils import pad_sents, pad_sents_char


class VocabEntry:
    """Vocabulary Entry, i.e. structure containing either src or tgt language terms."""

    def __init__(self, word2id=None):
        """Init a VocabEntry Instance.

        Args:
            word2id (dict): Dictionary mapping words to indices.
        """
        # Word-level representation
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()       # Dictionary mapping words to indices.
            self.word2id["<pad>"] = 0   # Pad Token
            self.word2id["<s>"] = 1     # Start Token
            self.word2id["</s>"] = 2    # End Token
            self.word2id["<unk>"] = 3   # Unknown Token

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.word_unk = self.word2id["<unk>"]
        self.word_pad = self.word2id["<pad>"]

        # Character-level representation
        self.char2id = dict()       # Dictionary mapping characters to indices.
        self.char2id["<p>"] = 0     # Pad token
        self.char2id["{"] = 1       # Start-of-word token
        self.char2id["}"] = 2       # End-of-word token
        self.char2id["<u>"] = 3     # Unknown token

        char_list = list(
            """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")
        for c in char_list:
            self.char2id[c] = len(self.char2id)

        self.id2char = {v: k for k, v in self.char2id.items()}
        self.char_pad = self.char2id["<p>"]
        self.char_unk = self.char2id["<u>"]
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]
        assert self.start_of_word + 1 == self.end_of_word

    def __getitem__(self, word):
        """Retrieve word's index. Return the index for the unk token if the word is out of
        vocabulary.
        """
        return self.word2id.get(word, self.word_unk)

    def __contains__(self, word):
        """Check if a word is captured by VocabEntry."""
        return word in self.word2id

    def __setitem__(self, key, value):
        """Raise error, if one tries to edit the VocabEntry."""
        raise ValueError("Vocabulary is read-only")

    def __len__(self):
        """Compute number of words in VocabEntry."""
        return len(self.word2id)

    def __repr__(self):
        """Representation of VocabEntry to be used when printing the object."""
        return "Vocabulary[size=%d]" % len(self)

    def id2word(self, wid):
        """Return mapping of index to word."""
        return self.id2word[wid]

    def add(self, word):
        """Add `word` to VocabEntry. Return the assigned index."""
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """Convert list of words or list of sentences of words into list or list of list
        of indices.

        Args:
            sents (List[str] or List[List[str]]): Sentence(s) in words.

        Returns:
            word_ids (List[int] or List[List[int]]): Sentence(s) in indices.
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """Convert a list of indices into a list words.

        Args:
            word_ids (List[int]): List of word ids.

        Returns:
            sents (List[str]): List of words.
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def words2charindices(self, sents):
        """Convert a list of sentences of words into a list of list of list of character
        indices.

        Args:
            sents (List[List[str]]): Sentence(s) in words.

        Returns:
            word_ids (List[List[List[int]]]): Sentence(s) in indices.
        """
        return [[[self.char2id.get(c, self.char_unk) for c in ("{" + w + "}")]
                                                     for w in s] for s in sents]

    def to_input_tensor(self, sents, device):
        """Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        Args:
            sents (List[List[str]]): List of sentences in words.
            device (torch.device): Device on which to load the tensor, i.e. CPU or GPU.

        Returns:
            sents_var (Tensor): Tensor of (max_sentence_length, batch_size).
        """
        word_ids = self.words2indices(sents)
        sents_padded = pad_sents(word_ids, self["<pad>"])
        sents_padded = torch.tensor(sents_padded, dtype=torch.long, device=device)
        return torch.transpose(sents_padded, 0, 1).contiguous()

    def to_input_tensor_char(self, sents, device):
        """Convert a list of sentences (words) into tensor with necessary padding.
        All words are padded to max_word_length of all words in the batch.
        All sentences are padded to max_sentence_length of all sentences in the batch.

        Args:
            sents (List[List[str]]): List of sentences in words.
            device (torch.device): Device on which to load the tensor, i.e. CPU or GPU.

        Returns:
            sents_var (Tensor): Tensor of (max_sent_length, batch_size, max_word_length).
        """
        char_ids = self.words2charindices(sents)
        sents_padded = pad_sents_char(char_ids, self.char_pad)
        sents_padded = torch.tensor(sents_padded, dtype=torch.long, device=device)

        return sents_padded.permute(1, 0, 2).contiguous()

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """Given a corpus construct a VocabEntry.

        Args:
            corpus (List[str]): Corpus of text produced by read_corpus function.
            size (int): Number of words in vocabulary.
            freq_cutoff (int, optional): If word occurs n < freq_cutoff times, drop the
                word. Defaults to 2.

        Returns:
            vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus.
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print("number of word types: {}, number of word types w/ frequency >= {}: {}"
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]

        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab:
    """Vocab encapsulating source and target langauges."""

    def __init__(self, src_vocab, tgt_vocab):
        """Init Vocab.

        Args:
            src_vocab (VocabEntry): VocabEntry for source language
            tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff):
        """Build Vocabulary.

        Args:
            src_sents (List[str]): Source sentences provided by read_corpus() function.
            tgt_sents (List[str]): Target sentences provided by read_corpus() function.
            vocab_size (int): Size of vocabulary for both source and target languages.
            freq_cutoff (int): If word occurs n < freq_cutoff times, drop the word.
        """
        assert len(src_sents) == len(tgt_sents)
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        return Vocab(src, tgt)

    def save(self, file_path):
        """Save Vocab to file as JSON dump."""
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id),
                  open(file_path, "w"), indent=2)

    @staticmethod
    def load(file_path):
        """Load vocabulary from JSON dump."""
        entry = json.load(open(file_path, "r"))
        src_word2id = entry["src_word2id"]
        tgt_word2id = entry["tgt_word2id"]
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """Representation of Vocab to be used when printing the object."""
        return "Vocab(source %d words, target %d words)" % (len(self.src), len(self.tgt))

#