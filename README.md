# DialoGPT
Implemented by `TensorFlow`

## Configuration

### Tool

`TensorFlow` for modeling  
`Hydra` for configurating  
`Weights & Bias` for monitoring

### Dataset
- [x] `감성 대화`: emotion coversation dataset, the number of total sentences is 27M.
- [ ] `한국어 SNS`: daily conversation SNS dataset, the number of total turns is 1,600M. 

In DialoGPT, The dataset comprises 147M dialogue instances, in total 1.8B words. 

(sourced by [AIHub](https://aihub.or.kr/aidata/30718)) 

### Tokenizer
`Subword`: We learn and apply *BPE* using the *SentencePiece* library. It prepends `'_'(U+2581)` to every word to mark the original whitespace, then tokenizes text into subword pieces.


`Morpeheme-aware Subword`: We apply *Mecab-Ko* and *BPE* in sequence to make morpheme-aware subwords. *BPE* is applied after the original text is split into morphemes. 

(referenced by [An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks](https://arxiv.org/abs/2010.02534))

### Model
`DialoGPT`: GPT-2 architecture.

(paper path [DIALOGPT : Large-Scale Generative Pre-training for Conversational Response GenerationialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances](https://arxiv.org/abs/1911.00536))

### Hyper parameter
- `small`: L=12, H=756, A=12
- `medium`: L=24, H=1024, A=12
- `large`: L=36, H=1280, A=12

We limit *dialogue length* to 250 words and *the number of conversation turn* to 5. 

We use `Adam` optimizer with learning rate as 2.5e^-4.

(referenced by [DIALOGPT : Large-Scale Generative Pre-training for Conversational Response GenerationialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances](https://arxiv.org/abs/1911.00536))

### Decoding Strategy

- [ ] Beam Search: It maintains a fixed-size set of partially-decoded sequences, called hypotheses. At each time step, beam search forms new hypotheses by appending each token in the vocabulary to each existing hypothesis, scoring the resulting sequences then selecting the highest scoring sequences.

<img src='./img/beam_search.png' width=500/>

(referenced by [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637))

(referenced by [DIVE INTO DEEP LEARNING](https://d2l.ai/chapter_recurrent-modern/beam-search.html))

### Evaluation
- [x] `PPL(perplexity)`:

<img src='./img/ppl.png' height=120/>

- [ ] `BLEU Score(Bilingual Evaluation Understudy Score)`: measures how many *n-grams* in a generated response overlap with those of the reference. 

<img src='./img/bleu.png'/>

- [ ] `NIST`: variant of `BLEU` that penalizes uniformative *n-grams* by assigning weights to *n-grams* according to their infromation gain.

- [ ] `SSA(Sensibleness and Specificity Average)`

(referenced by [DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances](https://arxiv.org/abs/2012.01775))

(referenced by [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977))


## Result

[click here](https://wandb.ai/gj98/DialoGPT/reports/DialoGPT--VmlldzoxNTY4NTE5?accessToken=z0wi83ax80m2utak0yqxasg0p52ccmsv1c7k33xnvmphsq7ru2asy819k9oik04x) for `Weights & Bias` reports.

`DialoGPT`
- `Accuarcy`: 0.4608
- `PPL(Perplexity)`: 20.02

**Example**

<img src='./img/result.png' width=400/>

