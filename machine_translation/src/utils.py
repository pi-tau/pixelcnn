"""Utility functions for preprocessing a corpus of data."""

import math

import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import torch

nltk.download("punkt")


#------------------------------ data preparation utilities ------------------------------#
def pad_sents_char(sents, char_pad_token):
    """Pad a list of sentences according to the longest sentence in the batch and the
    longest word in all sentences. The paddings are at the end of each word and at the end
    of each sentence.

    Args:
        sents (List[List[List[int]]]): List of sentences, where each sentence is
            represented as a list of words, and each word is represented as a list of characters.
            result of "words2charindices()" from "vocab.py".
        char_pad_token (int): Index of the character-padding token.

    Returns:
        sents_padded (List[List[List[int]]]): List of sentences where sentences/words
            shorter than the max length sentence/word are padded out with the appropriate
            pad token, such that each sentence in the batch now has same number of words
            and each word has an equal number of characters.
            output shape: (batch_size, max_sentence_length, max_word_length)
    """
    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s)
    max_sent_len = max(len(s) for s in sents)

    for s in sents:
        # Pad shorter words. Extend shorter sentences.
        s_pad = [[c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))] for w in s]
        s_pad.extend([[char_pad_token] * max_word_length] * max(0, max_sent_len - len(s_pad)))
        sents_padded.append(s_pad)

    return sents_padded

def pad_sents(sents, pad_token):
    """Pad a list of sentences according to the longest sentence in the batch. The
    paddings are at the end of each sentence.

    Args:
        sents (List[List[str]]): List of sentences, where each sentence is represented as
            a list of words.
        pad_token (str): Padding token.

    Returns:
        sents_padded (List[List[str]]): List of sentences where sentences shorter than the
            max length sentence are padded out with the pad_token, such that each sentence
            in the batch now has equal length.
    """
    sents_padded = []
    max_length = max(len(s) for s in sents)

    for s_ in sents:
        s = s_.copy()
        s.extend([pad_token] * max(0, max_length - len(s)))
        sents_padded.append(s)

    return sents_padded


#------------------------------ data generation utilities -------------------------------#
def read_corpus(file_path, source):
    """Read file, where each sentence is delineated by a "\n".

    Args:
        file_path (str): Path to file containing corpus.
        source (str): "src" or "tgt" indicating whether text is of the source language or
            target language.

    Returns:
        data (List[List(str)]): Sentences as a list of list of words.
    """
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            sent = word_tokenize(line)

            # only append <s> and </s> to the target sentence
            if source == "tgt":
                sent = ["<s>"] + sent + ["</s>"]
            data.append(sent)

    return data

def batch_iter(data, batch_size, shuffle=False):
    """Yield batches of source and target sentences reverse sorted by length
    (longest to shortest).

    Args:
        data (List[Tuple[List[str], List[str]]]): A list of tuples containing source and
            target sentence.
        batch_size (int): Batch size.
        shuffle (boolean, optional): Whether to randomly shuffle the dataset.
            Defaults to False.

    Yields:
        src_sents (List[List[str]]): A list of source sentences.
        tgt_sents (List[List[str]]): A list of target sentences.
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        yield src_sents, tgt_sents

def get_data(root, language):
    """Given root path to the folder containing the dataset, read the training, validation
    and test sets.

    Args:
        root (str): Root path to dataset folder.
        language (str): Source language. One of ("es", "de").

    Returns:
        src_train_sents (List[List[str]]): List of training source sentences.
        tgt_train_sents (List[List[str]]): List of training target sentences.
        src_dev_sents (List[List[str]]): List of development source sentences.
        tgt_dev_sents (List[List[str]]): List of development target sentences.
        src_test_sents (List[List[str]]): List of test source sentences.
        tgt_test_sents (List[List[str]]): List of test target sentences.
    """
    src_train_sents = read_corpus(root + "train.%s" % language, source="src")
    tgt_train_sents = read_corpus(root + "train.en", source="tgt")

    src_dev_sents = read_corpus(root + "dev.%s" % language, source="src")
    tgt_dev_sents = read_corpus(root + "dev.en", source="tgt")

    src_test_sents = read_corpus(root + "test.%s" % language, source="src")
    tgt_test_sents = read_corpus(root + "test.en", source="tgt")

    return (src_train_sents, tgt_train_sents,
            src_dev_sents, tgt_dev_sents,
            src_test_sents, tgt_test_sents)


#----------------------------- metrics evaluation utilities -----------------------------#
def eval_ppl(model, data, batch_size=64):
    """Evaluate the model perplexity on the given set.

    Args:
        model (NMT): NMT object.
        data (List[Tuple[List[str], List[str]]]): List of tuples containing src and tgt sentence.
        batch_size (int, optional): Size of the batch to evaluate perplexity on.
            Defaults to 64.

    Returns:
        ppl (float): Perplexity on the given sentences.
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.           # cumulative loss
    cum_tgt_words = 0.      # cumulative number of target words

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(data, batch_size):
            loss = model(src_sents, tgt_sents)

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading "<s>"
            cum_tgt_words += tgt_word_num_to_predict

    ppl = np.exp(cum_loss / cum_tgt_words)
    if was_training: model.train()
    return ppl

def compute_corpus_level_bleu_score(model, data, beam_size=2, max_decoding_time_step=10): # 5, 70
    """Evaluate the model corpus-level BLEU score on the given set.

    Args:
        model (NMT): Trained NMT model.
        data (Tuple(src_sent, tgt_sent)): Tuple containing source sentences and target sentences.
        beam_size (int, optional): Number of hypotheses to hold for a translation at every
            step. Defaults to 2.
        max_decoding_time_step (int, optional): Maximum sentence length that Beam search
            can produce. Defaults to 10.

    Returns:
        bleu_score (float): Corpus-level BLEU score.
    """
    was_training = model.training
    model.eval()

    source_sentences = data[0]      # Input sentences for translation.
    references = data[1]            # Gold-standard reference target sentences.

    if references[0][0] == "<s>":
        references = [ref[1:-1] for ref in references]

    # Run beam search to construct hypotheses for a list of src-language sentences.
    hypotheses = []
    with torch.no_grad():
        for src_sent in source_sentences:
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    # Compute the corpsus-level BLEU score.
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in top_hypotheses])
    if was_training: model.train()
    return bleu_score


#----------------------------------- helper functions -----------------------------------#
def fix_random_seeds(seed):
    """Manually set the seed for random number generation.
    Also set CuDNN flags for reproducible results using deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


#