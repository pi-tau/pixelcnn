"""Neural Machine Translation (NMT) model.
The model uses a sequence-to-sequence with attention architecture.
It involves two LSTM recurrent neural networks - Encoder and Decoder.
The Encoder encodes the source sentence and provides initial hidden state and
initial cell state for the Decoder.
The Decoder generates the target sentence one step at a time. On each step
the Decoder uses Attention to focus on a particular part of the source sentence.

Given a sentence in the source language, we look up the word embeddings from
the ModelEmbeddings object.
The word embeddings of the source sentence are fed to the bidirectional encoder,
yielding hidden states and cell states for both the forward and backward LSTMs.
The forward and backward versions are concatenated to give hidden and cell states.
The decoder's first hidden state and first cell state are initialized with a
linear projection of the encoder's final hidden state and final cell state.
At each step we use the decoder hidden state to compute multiplicative attention
over the encoder hidden states (concatenated from the forward and backward pass)
The attention output (2h) is concatenated with the decoder hidden state (h) and
passed through a linear layer, tanh, and dropout to attain the combined-output vector.
At the end of each step we produce a probability distribution over the target words.

To train the network we compute the softmax cross-entropy loss between the produced
probability distribution and the one-hot vector of the target word at each timestep.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from src.char_decoder import CharDecoder
from src.model_embeddings import ModelEmbeddings


Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class NMT(nn.Module):
    """Simple Neural Machine Translation Model:
    
    Attributes:
        hidden_size (int): The size of the hidden states of the recurrent neural networks.
        dropout_rate (float): Dropout probability for attention.
        vocab (Vocab): Vocabulary object containing src and tgt languages.
        model_embeddings_source (ModelEmbeddings): Embedding layer for source language.
        model_embeddings_target (ModelEmbeddings): Embedding layer for target language.
        encoder (nn.LSTM): One-layer bidirectional LSTM network.
        decoder (nn.LSTMCell): One-layer unidirectional LSTM cell.
        h_projection (nn.Linear): Linear layer used to compute the initial hidden state of
            the decoder from the final hidden state of the encoder.
        c_projection (nn.Linear): Linear layer used to compute the initial cell state of
            the decoder from the final cell state of the encoder.
        att_projection (nn.Linear): Linear layer used to project the encoder hidden states
            before computing multiplicative attention score.
        combined_output_projection (nn.Linear): Linear layer used to project the attention
            output to compute the decoder output.
        dropout (nn.Dropout): Dropout layer applied on the decoder output.
        target_vocab_projection (nn.Linear): Linear layer producing scores over the target
            vocabulary from the decoder output.
        charDecoder (CharDecoder): A CharDecoder object used to replace <UNK> tokens
            with words generated one character at a time.
    """

    def __init__(self, word_embed_size, char_embed_size, hidden_size, vocab,
                 dropout_rate=0.2, kernel_size=5, padding=1):
        """Init NMT Model.

        Args:
            word_embed_size (int): Embedding size for the words.
            char_embed_size (int): Embedding size for the characters.
            hidden_size (int): Hidden Size, the size of hidden states (dimensionality).
            vocab (Vocab): Vocabulary object containing src and tgt languages
                See vocab.py for documentation.
            dropout_rate (float, optional): Dropout probability, for attention.
                Defaults to 0.2.
            kernel_size (int, optional): Kernel size for the character-level CNN.
                Defaults to 5.
            padding (int, optional): Size of padding for the character-level CNN.
                Defaults to 1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # Initialization of the model architecture.
        # Model Embeddings.
        self.model_embeddings_source = ModelEmbeddings(word_embed_size, char_embed_size,
            vocab.src, kernel_size=kernel_size, padding=padding, dropout_rate=dropout_rate)
        self.model_embeddings_target = ModelEmbeddings(word_embed_size, char_embed_size,
            vocab.tgt, kernel_size=kernel_size, padding=padding, dropout_rate=dropout_rate)

        # Sequence-to-Sequence with attention architecture.
        self.encoder = nn.LSTM(word_embed_size, hidden_size, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(word_embed_size + hidden_size, hidden_size, bias=True)
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # Character decoder.
        self.charDecoder = CharDecoder(hidden_size=hidden_size, char_embed_size=char_embed_size,
                                       target_vocab=vocab.tgt)

        # Initialize the model parameters.
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
            # if len(param.shape) > 1:
            #     torch.nn.init.kaiming_normal_(param)
            # else:
            #     torch.nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, source, target):
        """Take a mini-batch of source and target sentences and compute the log-likelihood
        of target sentences under the language model learned by the NMT system.

        Run a forward pass on the network:
            1. Run the input `source_padded` through the encoder.
            2. Generate sentence masks for `source_padded`.
            3. Apply the decoder to compute the decoder outputs.
            4. Compute log-probability distribution over the target vocabulary from the
                decoder outputs.
            5. Compute the word-level loss.

            6. Use the characters from the target sentence to train the character-level
                decoder.
            7. Compute the oov loss (out-of-vocabulary) and add it to the word-level loss.

        Args:
            source (List[List[str]]): List of source sentence tokens.
            target (List[List[str]]): List of target sentence tokens. The target sentences
                must be wrapped by `<s>` and `</s>`.

        Returns:
            scores (Tensor): Tensor of shape (b,) representing the log-likelihood of
                generating the gold-standard target sentence for each example in the input
                batch. Here b = batch size.
        """
        # Compute sentence lengths.
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors.
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)
        target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device=self.device)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        # Compute the scores.
        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)
        word_scores = self.target_vocab_projection(combined_outputs)

        # Compute the word-level loss
        loss = F.cross_entropy(word_scores.view(-1, len(self.vocab.tgt)),
            target_padded[1:].view(-1), ignore_index=self.vocab.tgt.word_pad, reduction="sum")

        # Generate input char sequence and target output char sequence for training char-level decoder.
        max_word_len = target_padded_chars.shape[-1]
        target_chars_oov = target_padded_chars[1:].contiguous().view(-1, max_word_len)  # (tgt_len * b, w)
        target_chars_oov = target_chars_oov.transpose(1, 0).contiguous()                # (w, tgt_len * b)
        in_seq = target_chars_oov[ : -1]
        out_seq = target_chars_oov[1 : ]

        # Use the combined_outputs to initialize the char-level decoder hidden and cell states.
        rnn_hidden_oov = combined_outputs.view(-1, self.hidden_size).unsqueeze(0)       # (1, tgt_len * b, h)
        rnn_cell_oov = torch.zeros(rnn_hidden_oov.shape, device=self.device)

        # Compute character-based decoder loss and add to the word-based decoder loss.
        char_scores, _ = self.charDecoder(in_seq, (rnn_hidden_oov, rnn_cell_oov))
        oovs_losses = F.cross_entropy(char_scores.permute(0, 2, 1), out_seq,
                                      ignore_index=self.vocab.tgt.char_pad, reduction="sum")
        loss = loss + 1.0 * oovs_losses

        return loss

    def encode(self, source_padded, source_lengths):
        """Apply the encoder to source sentences to obtain encoder hidden states.
        Additionally, take the final states of the encoder and project them to obtain
        initial states for decoder.

        Run the source sentences through the encoder:
            1. Construct a tensor `X` of source sentences with shape (src_len, b, e) using
                the source model embeddings.
            2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
                - before we can apply the encoder, we need to apply the
                  `pack_padded_sequence` function to `X`
                - after we apply the encoder, we need to apply the `pad_packed_sequence`
                  function to `enc_hiddens`
                - the shape of the tensor returned by the encoder is (src_len, b, h*2) and
                  we want to return a tensor of shape (b, src_len, h*2) as `enc_hiddens`
            3. Compute the initial hidden state of the decoder by applying `h_projection`
                layer to `last_hidden`. Compute the initial cell state of the decoder by
                applying `c_projection` layer to `last_cell`.

        Args:
            source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b),
                where b = batch_size, src_len = maximum source sentence length. Note that
                these have already been sorted in order of longest to shortest sentence.
            source_lengths (List[int]): List of actual lengths for each of the source
                sentences in the batch.

        Returns:
            enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.
            dec_init_state (Tuple(Tensor, Tensor)): Tuple of tensors representing the
                initial hidden state and the initial cell state of the decoder.
        """
        X = self.model_embeddings_source(source_padded)
        packed_X = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(packed_X)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True)

        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)

        dec_init_hidden = self.h_projection(last_hidden)
        dec_init_cell = self.c_projection(last_cell)

        dec_init_state = (dec_init_hidden, dec_init_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
        """Compute the decoder output vectors for a batch.
        Given the encoder hidden states, the decoder initial state and a batch of target
        sentences, copute the decoder output vectors.
        At each step the input to the decoder is the next item from the target sequence.
        This approach is known as `teacher forcing`.

        Decode the target sentence:
            1. Apply the attention projection layer to `enc_hiddens` to obtain
                `enc_hiddens_proj`, which should be of shape (b, src_len, h),
            2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using
                the target model embeddings.
            3. Use the torch.split function to iterate over the time dimension of Y.
                Within the loop, this will give us Y_t of shape (1, b, e)
                - Squeeze Y_t into a tensor of dimension (b, e). 
                - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
                - Use the step function to compute the the Decoder's next (cell, state)
                  values as well as the new combined output o_t.
                - Append o_t to combined_outputs.
                - Update o_prev to the new o_t.
            4. Use torch.stack to convert combined_outputs from a list of length tgt_len
                of tensors shape (b, h), to a single tensor of shape (tgt_len, b, h).

        Args:
            enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                b = batch size, src_len = maximum source sentence length.
            dec_init_state (Tuple(Tensor, Tensor)): Initial state and cell for decoder.
            target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b),
                where tgt_len = maximum target sentence length, b = batch size.

        Returns:
            combined_outputs (Tensor): combined output tensor (tgt_len, b, h), where
                tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell).
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero.
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step.
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings_target(target_padded)

        for Y_t in torch.split(Y, 1, dim=0):
            # When using the squeeze() function we have to make sure to specify the dimension we want to
            # squeeze over. Otherwise, we will remove the batch dimension accidentally, if batch_size = 1.
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.cat((Y_t, o_prev), dim=-1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs

    def step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        """Compute one forward step of the LSTM decoder, including the attention computation.

        Run a single timestep of the Decoder:
            1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
            2. Split dec_state into its two parts (dec_hidden, dec_cell)
            3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
                Note: b = batch_size, src_len = maximum source length, h = hidden size.
            4. Use enc_masks to set the attention score to "-inf" for the padding.
            5. Apply softmax to e_t to yield alpha_t - the attention distribution. 
            6. Compute the attention output vector, a_t, of shape (b, 2h). The attention
                output vector is the weighted sum of enc_hiddens (b, src_len, 2h).
            7. Concatenate dec_hidden with a_t to compute tensor U_t.
            8. Apply the combined output projection layer to U_t to compute tensor V_t.
            9. Compute tensor o_t by first applying tanh function and then the dropout layer.

        Args:
            Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h).
                The input for the decoder, where b = batch size, e = embedding size, h = hidden size.
            dec_state (Tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h),
                where b = batch size, h = hidden size. First tensor is decoder's prev
                hidden state, second tensor is decoder's prev cell.
            enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2),
                where b = batch size, src_len = maximum source length, h = hidden size.
            enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h.
                Tensor is with shape (b, src_len, h), where b = batch size,
                src_len = maximum source length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                where b = batch size, src_len is maximum source length.

        Returns:
            dec_state (Tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h),
                where b = batch size, h = hidden size. First tensor is decoder's new
                hidden state, second tensor is decoder's new cell state.
            o_t (Tensor): Combined output Tensor at timestep t, shape (b, h), where
                b = batch size, h = hidden size.
            e_t (Tensor): Attention scores distribution Tensor at timestep t, shape (b, src_len).
        """
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state

        # torch.bmm - batch matrix-matrix product of matrices
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float("inf"))

        alpha_t = F.softmax(e_t, dim=-1)
        a_t = torch.bmm(alpha_t.unsqueeze(dim=1), enc_hiddens).squeeze(dim=1)
        U_t = torch.cat((dec_hidden, a_t), dim=-1)
        V_t = self.combined_output_projection(U_t)
        o_t = self.dropout(torch.tanh(V_t))
        # o_t = self.dropout(F.tanh(V_t)) # deprecated

        return dec_state, o_t, e_t

    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """Generate sentence masks for encoder hidden states.

        Args:
            enc_hiddens (Tensor): Encodings of shape (b, src_len, 2*h), where
                b = batch size, src_len = max source length, h = hidden size. 
            source_lengths (List[int]): List of actual lengths for each of the sentences
                in the batch.

        Returns:
            enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len), where
                b = batch size, src_len is maximum source length.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def greedy_decoding(self, src_sent, max_length=50):
        """Given a single source sentence, decode the sentence, yielding translation in
        the target language.

        Args:
            src_sent (List[str]): A single source sentence (list of words).
            max_length (int, optional): Maximum number of time steps to unroll the
                decoding RNN. Defaults to 50.

        Returns:
            hypothesis (List[str]): The target sentence represented as a list of words.
        """
        x = self.vocab.src.to_input_tensor_char([src_sent], self.device)
        enc_hiddens, dec_init_state = self.encode(x, [len(src_sent)])
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        dec_state = dec_init_state
        o_prev = torch.zeros(1, self.hidden_size, device=self.device)

        hypothesis = ["<s>"]
        while len(hypothesis) < max_length:
            y_t = self.vocab.tgt.to_input_tensor_char([[hypothesis[-1]]], device=self.device)
            y_t_embed = self.model_embeddings_target(y_t)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)
            Ybar_t = torch.cat((y_t_embed, o_prev), dim=-1)

            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj,
                                          enc_masks=None)
            log_p_t = F.log_softmax(self.target_vocab_projection(o_t), dim=-1)
            max_elem, max_idx = torch.max(log_p_t, dim=1)
            next_word = self.vocab.tgt.id2word[max_idx.item()]

            # If the decoded word is "<unk>", use the character decoder to infer an
            # out-of-vocabulary word.
            # if next_word == self.vocab.tgt.word2id[self.vocab.tgt.word_unk]:
            if next_word == "<unk>":
                next_word = self.charDecoder.decode_greedy(
                    (dec_state[0].unsqueeze(0), dec_state[1].unsqueeze(0)),
                    device=self.device
                )[0]
            
            # If the decoded word is "</s>", stop the inference and return the translated
            # sentence.
            if next_word == "</s>":
                break

            hypothesis.append(next_word)
            o_prev = o_t

        return hypothesis[1:]

    @property
    def device(self):
        """device: Determine which device to place the Tensors upon, CPU or GPU."""
        return self.att_projection.weight.device

    @staticmethod
    def load(model_path):
        """Load the model from a file."""
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["args"]
        model = NMT(vocab=params["vocab"], **kwargs)
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, path):
        """Save the model to a file."""
        params = {
            "args": dict(word_embed_size=self.model_embeddings_source.word_embed_size,
                         char_embed_size=self.model_embeddings_source.char_embed_size,
                         hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            "vocab": self.vocab,
            "state_dict": self.state_dict()
        }
        torch.save(params, path)


    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """Given a single source sentence, perform beam search, yielding translations in
        the target language.

        Args:
            src_sent (List[str]): a single source sentence (words)
            beam_size (int, optional): Beam size. Defaults to 5.
            max_decoding_time_step (int, optional): Maximum number of time steps to unroll
                the decoding RNN for. Defaults to 70.

        Returns:
            hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor_char([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                        src_encodings_att_linear.size(1),
                                                        src_encodings_att_linear.size(2))

            y_tm1 = self.vocab.tgt.to_input_tensor_char(list([hyp[-1]]
                                            for hyp in hypotheses), device=self.device)
            y_t_embed = self.model_embeddings_target(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings,
                                            exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode="trunc")
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            decoderStatesForUNKsHere = []
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]

                # Record output layer in case UNK was generated
                if hyp_word == "<unk>":
                    hyp_word = "<unk>" + str(len(decoderStatesForUNKsHere))
                    decoderStatesForUNKsHere.append(att_t[prev_hyp_id])

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(decoderStatesForUNKsHere) > 0 and self.charDecoder is not None:  # decode UNKs
                decoderStatesForUNKsHere = torch.stack(decoderStatesForUNKsHere, dim=0)
                decodedWords = self.charDecoder.decode_greedy(
                    (decoderStatesForUNKsHere.unsqueeze(0), decoderStatesForUNKsHere.unsqueeze(0)),
                    max_length=21,
                    device=self.device
                )
                assert len(decodedWords) == decoderStatesForUNKsHere.size()[0], "Incorrect number of decoded words"
                for hyp in new_hypotheses:
                    if hyp[-1].startswith("<unk>"):
                        hyp[-1] = decodedWords[int(hyp[-1][5:])]  # [:-1]

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses

#