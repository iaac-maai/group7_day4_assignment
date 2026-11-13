---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - pytorch
  - built-environment
  - iaac-maai
  - ai-methods-1
  - concrete
  - crack-detection
metrics:
  - accuracy
  - recall
---

# day4-group7-crack-classifier: crack / no crack in concrete photos

This model tells whether a close-up photo of a concrete surface shows a **crack** or **no crack**, and gives the probability of each. It is meant to help a facility manager or inspection engineer sort a large set of inspection photos, so that the photos most likely to show a crack are checked first. It is a **ResNet18** (11.2 million parameters, 1,026 trainable) pretrained on ImageNet, with its layers frozen and a new last layer trained on 2,800 images of the METU Concrete Crack dataset. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/group7_day4_assignment
- **Team:** Ugo, Alvaro, Sneha Shenoy

## Model at a glance

![What the model does](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/predictions.png)

*Test images with the true class, the predicted class and P(crack); red titles are mistakes. The model gets 598 of the 600 test images right.*

| Model | Accuracy | Crack recall | Crack precision |
|---|---|---|---|
| Baseline: always predict `crack` | 50.0% | 100.0% | 50.0% |
| **This model, test set** | **99.7%** | **99.7%** | **99.7%** |
| This model, 21 new web images | 86% | 80% | 100% |

![New images](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/own_photos_ranking.png)

*On 21 images from outside the dataset, every wide crack is found and no plain surface is flagged, but 3 hairline cracks in wider photos are missed (red). Expect lower performance than the test score on your own photos.*

## Intended use

- **Intended users and use:** inspectors and facility managers who want to **sort** close-up photos of concrete surfaces by how likely they are to show a crack, to decide what to look at first.
- **Out-of-scope uses:** structural safety decisions without an engineer; judging how severe or dangerous a crack is; locating a crack in a photo; whole-facade, aerial or drone photos; other materials (brick, plaster, tiles, painted surfaces); any decision about people.

## Model details

| | |
|---|---|
| **Task** | Image classification (2 classes) |
| **Architecture** | ResNet18 with a new last layer `nn.Linear(512, 2)` |
| **Parameters** | 11,177,538 (1,026 trainable; all others frozen) |
| **Base model** | `torchvision` ResNet18, `ResNet18_Weights.DEFAULT`, pretrained on ImageNet |
| **Input** | One RGB image, preprocessed with `ResNet18_Weights.DEFAULT.transforms()`: resized to 256, centre-cropped to 224 × 224, normalised with ImageNet statistics. **Use the same transforms at prediction time.** |
| **Output** | Two logits; apply softmax to get P(crack) and P(no crack). Recommended decision: crack if P(crack) ≥ 0.5 |
| **Files** | `model.pt` (`state_dict`, 44.8 MB) |
| **Library versions** | Python 3.11, torch 2.15.0, torchvision 0.29.2 |

### Classes

| ID | Class | Definition |
|---|---|---|
| 0 | `crack` | The image shows at least one visible crack in the concrete surface (dataset folder `Positive`) |
| 1 | `no_crack` | Intact concrete surface: pores, stains and small surface marks are not cracks (dataset folder `Negative`) |

## Training data

| | |
|---|---|
| **Dataset** | Concrete Crack Images for Classification (METU) |
| **Source** | [Mendeley Data, DOI 10.17632/5y9wdsg2zt.2](https://data.mendeley.com/datasets/5y9wdsg2zt) (Özgenel, 2019); Kaggle mirror `arunrk7/surface-crack-detection` |
| **Licence** | CC BY 4.0 |
| **Size** | Random sample of 4,000 of the 40,000 images (2,000 per class): train 2,800 / validation 600 / test 600, stratified by class, `random_state=42` |
| **Coverage** | 227 × 227 px close-up patches cut from 458 photos of concrete surfaces on METU campus buildings (Ankara, Turkey). Most cracks are wide, dark and fill a large part of the patch |

## Training procedure

- **Settings:** Adam, learning rate 1e-3 (new layer only), batch size 32, cross-entropy loss, up to 15 epochs with early stopping on the validation loss (patience 3).
- **Augmentation and regularisation:** random resized crop (scale 0.7–1.0), horizontal and vertical flips, rotations by 0/90/180/270°, brightness and contrast jitter; frozen backbone with BatchNorm kept in evaluation mode.
- **Hardware and time:** MacBook Pro, Apple Silicon (`mps`), 2.7 minutes.
- **Model selection:** weights with the lowest validation loss (epoch 15). An experiment that also fine-tuned `layer4` (learning rate 1e-4) was worse on validation (99.5% accuracy, 99.3% crack recall vs. 99.7% and 100%) and was not kept. The test set was used once, at the end.

![Training curves](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/training_curves.png)

*The validation loss keeps falling, but validation accuracy is flat at 99.7% from epoch 6. Training and validation stay close: no overfitting.*

## Evaluation

| Metric (test, n = 600) | Baseline (always `crack`) | This model |
|---|---|---|
| Accuracy | 50.0% | **99.7%** |
| Crack precision | 50.0% | **99.7%** |
| Crack recall | 100.0% | **99.7%** |

**Per class:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| crack | 0.997 | 0.997 | 0.997 |
| no_crack | 0.997 | 0.997 | 0.997 |

![Confusion matrix](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/confusion_matrix.png)

![Model vs. baseline](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/model_vs_baseline.png)

**Where it fails:** all mistakes involve **thin, faint lines**. On the test set it missed one thin, low-contrast crack next to a dark hole (75% "no crack") and flagged one faint scratch (62% "crack"). On validation it flagged a wire lying on the surface (97% "crack"). On 21 new web images it missed 3 hairline cracks photographed from further away (P(crack) 49%, 18%, 2%).

![Failure cases](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/failures.png)

### Choosing a decision threshold

![Threshold sweep](https://huggingface.co/iaac-maai/day4-group7-crack-classifier/resolve/main/threshold_sweep.png)

On validation, crack recall stays at 100% for every threshold from 0.05 to 0.6, and precision is stable at 99.3% between 0.3 and 0.7. We recommend **0.5**, in the middle of that range. Because a missed crack costs much more than a false alarm, users working with wider or lower-quality photos may prefer **0.3**: on our 21 new images it would catch one more hairline crack without any false alarm.

## Limitations and biases

- **The test score is probably optimistic.** The 40,000 patches come from only 458 photos, and the dataset does not record which patch comes from which photo, so patches of the same wall can be in both training and test. On new images accuracy fell to 86% (n = 21).
- **Hairline cracks and scale.** The model learned "wide dark line = crack". It misses thin cracks, especially when the photo shows a large area, because they nearly disappear at 224 × 224 pixels. Use close-ups, or cut large photos into tiles.
- **Dark lines that are not cracks** (wires, scratches, joints, shadows) can be flagged as cracks.
- **Narrow data:** bare concrete from one campus in Turkey. Not tested on plaster, brick, painted or wet surfaces, other climates, or drone images.
- **Label noise:** at least one dataset image labelled "no crack" appears to show a real crack.
- **No location, no severity:** it classifies the whole image only.

## How to use

```python
import torch
from torch import nn
from torchvision import models
from PIL import Image
from huggingface_hub import hf_hub_download

CLASSES = ["crack", "no_crack"]                            # output order
model = models.resnet18()                                  # same architecture as in training
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))   # new 2-output last layer
model.load_state_dict(torch.load(hf_hub_download("iaac-maai/day4-group7-crack-classifier", "model.pt"),
                                 map_location="cpu"))
model.eval()

transform = models.ResNet18_Weights.DEFAULT.transforms()   # the same preprocessing as in training
image = Image.open("wall.jpg").convert("RGB")
with torch.no_grad():
    probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
print(dict(zip(CLASSES, probabilities.tolist())))
print("CHECK" if probabilities[0] >= 0.5 else "ok")         # recommended threshold on P(crack)
```

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/group7_day4_assignment. Run `notebook/neural_network.ipynb` with the packages in `requirements.txt` (seed 42 everywhere).

## Authors and licence

Ugo, Alvaro, Sneha Shenoy. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is subject to its own licence: CC BY 4.0 (Özgenel, Ç. F. (2019), Concrete Crack Images for Classification, Mendeley Data, v2).
