# VisionScribe AI

Image captioning with a **CNN encoder and LSTM decoder** in PyTorch, trained on Flickr8k. Upload a photo and get a one-sentence description, optionally read aloud, which makes it useful for accessibility.

## How it works

```
image ─► DenseNet-121 (ImageNet-pretrained) ─► 256-d image embedding ─┐
                                                                     ▼
                         <start> a dog runs ... ─► word embeddings ─► LSTM (512) ─► next-word softmax
```

- **Encoder:** DenseNet-121 with global average pooling, projected to a 256-d embedding. A ResNet-50 encoder is included as a baseline for comparison.
- **Decoder:** single-layer LSTM. The image embedding is fed as the first timestep, and the model is trained with teacher forcing and cross-entropy loss.
- **Inference:** greedy decoding or beam search (default beam = 3).
- **Evaluation:** corpus BLEU-1 to BLEU-4 against all 5 reference captions per image, on a held-out test split of 1,000 images.
- **Training:** AdamW with gradient clipping, LR halved when validation BLEU-4 plateaus, and early stopping on validation BLEU-4.

## Results

Fill this table in from `results/test_bleu.json` after running the steps below.

| Encoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| ResNet-50 (baseline) | – | – | – | – |
| DenseNet-121 | – | – | – | – |

## Quick start

```bash
pip install -r requirements.txt
```

**1. Get the data.** Download [Flickr8k from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and unzip it so that you have:

```
data/Flickr8k/Images/*.jpg
data/Flickr8k/captions.txt
```

**2. Train both encoders.** A GPU is recommended; Google Colab's free T4 works.

```bash
python train.py --data data/Flickr8k --backbone densenet121 --out runs/densenet121
python train.py --data data/Flickr8k --backbone resnet50    --out runs/resnet50
```

**3. Evaluate on the test split**

```bash
python evaluate.py --data data/Flickr8k --runs runs/densenet121 runs/resnet50 --beam 3
```

**4. Caption your own images**

```bash
python caption.py --run runs/densenet121 my_photo.jpg
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
- Fine-tune the last DenseNet block with a lower learning rate
- Train on Flickr30k or COCO for a larger vocabulary
