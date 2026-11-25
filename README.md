# VisionScribe AI

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khush-ahuja/VisionScribe-AI/blob/main/colab_train.ipynb)

Image captioning with a **CNN encoder and LSTM decoder** in PyTorch, trained on Flickr8k. Upload a photo and get a one-sentence description, optionally read aloud, which makes it useful for accessibility.

## How it works

```
image ─► ResNet-50 / DenseNet-121 (ImageNet-pretrained) ─► 256-d embedding ─┐
                                                                     ▼
                         <start> a dog runs ... ─► word embeddings ─► LSTM (512) ─► next-word softmax
```

- **Encoder:** a frozen ImageNet-pretrained CNN (ResNet-50 or DenseNet-121) with global average pooling, projected to a 256-d embedding. Both encoders are trained and compared under identical settings.
- **Decoder:** single-layer LSTM. The image embedding is fed as the first timestep, and the model is trained with teacher forcing and cross-entropy loss.
- **Inference:** greedy decoding or beam search (default beam = 3).
- **Evaluation:** corpus BLEU-1 to BLEU-4 against all 5 reference captions per image, on a held-out test split of 1,000 images.
- **Training:** AdamW with gradient clipping, LR halved when validation BLEU-4 plateaus, and early stopping on validation BLEU-4.

## Results

Held-out Flickr8k test split (1,000 images, 5 reference captions each), beam size 3. Both models use the same LSTM decoder, vocabulary, training budget (12 epochs, early stopping on validation BLEU-4) and a single T4 GPU.

| Encoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| **ResNet-50** | **0.611** | **0.433** | **0.298** | **0.207** |
| DenseNet-121 | 0.585 | 0.405 | 0.276 | 0.189 |

ResNet-50 scored **9.5% higher BLEU-4** than DenseNet-121, so it is the default model. A likely reason: with a frozen encoder that passes a single pooled vector to the decoder, ResNet-50's 2048-d features carry more information than DenseNet-121's 1024-d ones. Reproduce with the Colab notebook above.

## Quick start

```bash
pip install -r requirements.txt
```

**1. Get the data.** Download [Flickr8k from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and unzip it so that you have:

```
data/Flickr8k/Images/*.jpg
data/Flickr8k/captions.txt
```

**2. Train both encoders.** A GPU is recommended. The easiest route is the Colab notebook above: it downloads the data, trains both models and prints the BLEU table.

```bash
python train.py --data data/Flickr8k --backbone resnet50    --out runs/resnet50
python train.py --data data/Flickr8k --backbone densenet121 --out runs/densenet121
```

**3. Evaluate on the test split**

```bash
python evaluate.py --data data/Flickr8k --runs runs/resnet50 runs/densenet121 --beam 3
```

**4. Caption your own images**

```bash
python caption.py --run runs/resnet50 my_photo.jpg
streamlit run app.py
```

## Project structure

```
visionscribe/
  model.py     CNN encoder, LSTM decoder, greedy and beam search
  data.py      Flickr8k loading, deterministic 6000/1000/1000 split, transforms
  vocab.py     tokenizer and vocabulary
  metrics.py   corpus BLEU-1..4
train.py       training loop with validation BLEU and early stopping
evaluate.py    test-set BLEU and encoder comparison
caption.py     command-line captioning
app.py         Streamlit demo with optional text-to-speech
```

## Possible next steps

- Add soft attention over spatial CNN features (Show, Attend and Tell)
- Fine-tune the last encoder block with a lower learning rate
- Train on Flickr30k or COCO for a larger vocabulary
