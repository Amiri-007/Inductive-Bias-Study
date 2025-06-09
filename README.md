# MNIST Inductive-Bias Study: **CNN vs FCNN**

*Visualising how architecture shapes learning dynamics*

> **Course**: Intro to Deep Learning (Spring 2025)  
> **Author**: *Your Name*
> **Repo**: [https://github.com/your-handle/mnist-cnn-vs-fcnn](https://github.com/your-handle/mnist-cnn-vs-fcnn)

---

## 1 Project Purpose

Classical MNIST benchmarks show many models can reach > 99 % accuracy, but **how** they arrive there differs.
This mini-project isolates one design choice—**convolutional inductive bias**—by:

1. Building two equal-capacity (< 20 k params) networks

   * **LeNet-2D CNN** (locality + translation invariance)
   * **FCNN-2D** (bias-free, LayerNorm)
2. Forcing both to compress every image into a **2-D latent code**.
3. Recording that latent space every 100 mini-batches → MP4 “evolution” videos.
4. Quantitatively comparing accuracy / AUROC / F1 / MCC and qualitatively analysing cluster geometry + decision wedges.

The deliverables show **why CNNs learn faster** (spoke-like clusters within one epoch) and how a bias-free network eventually converges but with very different geometry.

---

## 2 Repository Structure

```
.
├── notebook/                # Single Jupyter notebook (CNN & FCNN)
│   └── bias_demo.ipynb
├── src/                     # Clean script versions
│   ├── models.py            # LeNet2D, FCNN2D classes
│   ├── train.py             # Training + snapshot recording
│   └── video_utils.py       # create_embedding_video()
├── videos/
│   ├── cnn_embedding_evolution.mp4
│   └── fcnn_embedding_evolution.mp4
├── figures/
│   ├── cnn_regions.png
│   ├── fcnn_regions_zoomed.png
│   └── final_embeddings_compare1.png
├── requirements.txt
└── README.md                # ← you are here
```

> *Notebook-only mode*: just open **`notebook/bias_demo.ipynb`** and run top-to-bottom; all figures and videos regenerate automatically.

---

## 3 Quick Start

```bash
# clone & set up environment
git clone https://github.com/your-handle/mnist-cnn-vs-fcnn.git
cd mnist-cnn-vs-fcnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# launch notebook
jupyter lab notebook/bias_demo.ipynb
```

or run the clean script pipeline:

```bash
python -m src.train  --batch 128 --epochs 30 --snapshot 100
# outputs MP4 videos + PNG decision plots in ./videos & ./figures
```

---

## 4 Key Results

| Model              | Params |   Test Acc |  Macro-F1 | Macro-AUROC |       MCC |
| ------------------ | -----: | ---------: | --------: | ----------: | --------: |
| **CNN (LeNet-2D)** | 19 180 | **97.7 %** | **0.977** |   **0.999** | **0.977** |
| **FCNN-2D**        | 19 304 |     94.0 % |     0.945 |       0.994 |     0.939 |

* CNN forms radial spokes in 2-D after ≤ 1 epoch; FCNN’s latent remains a dense blob until \~epoch 8.
* Decision wedges illustrate CNN boundaries hug clusters, FCNN extrapolates far outside data.
* Videos (see `videos/`) visually narrate the untangling process.

---

## 5 How to Reproduce

1. **Change hyper-parameters**

   ```bash
   python -m src.train --epochs 50 --snapshot 50 --no-cuda
   ```

   *snapshot frequency* controls video smoothness; more epochs create longer movies.

2. **Regenerate decision-region plots**

   ```python
   from src.models import LeNet2D, FCNN2D
   from src.video_utils import plot_embedding_with_regions
   # assumes embeddings saved as numpy arrays
   plot_embedding_with_regions(cnn_model, emb_cnn, lbl_cnn,
                               title="CNN decision regions",
                               save="figures/cnn_regions.png")
   ```

3. **Save / load checkpoints**
   `train.py` saves `cnn_last.pth` & `fcnn_last.pth`; reload with

   ```python
   cnn = LeNet2D(); cnn.load_state_dict(torch.load("checkpoints/cnn_last.pth"))
   ```

---

## 6 Core Files Explained

| File              | What it does                                                                            |
| ----------------- | --------------------------------------------------------------------------------------- |
| `models.py`       | Defines **`LeNet2D`** (2 conv + pool → FC → 2-D) and **`FCNN2D`** (784→24→16→2).        |
| `train.py`        | Full training loop: metrics, early snapshots, epoch logging.                            |
| `video_utils.py`  | `create_embedding_video()` turns snapshot list → MP4 (uses Matplotlib + `imageio`).     |
| `bias_demo.ipynb` | One-shot interactive notebook: trains both nets, renders videos & plots, prints tables. |

---

## 7 Dependencies

* Python 3.9+
* PyTorch ≥ 2.0 • torchvision
* NumPy • pandas • scikit-learn
* Matplotlib • imageio • tqdm

Install via `pip install -r requirements.txt` (CPU-only wheels available).

---

## 8 Learning Take-aways

* **Inductive bias** (local conv filters, weight sharing, pooling) dramatically accelerates class disentanglement in latent space.
* Forcing a **2-D bottleneck** is an effective pedagogical tool: lets you watch clusters form and see how linear heads carve decision wedges.
* Equal parameter budgets are essential for fair comparisons; here both models have ≈ 19 k trainable weights.
* Latent evolution videos are a powerful communication aid—professor feedback highlighted the clarity of this visual approach.

---

## 9 Credits & License

Project by *Your Name* for NYU **Intro to Deep Learning** (Prof. Alfredo Canziani).
MIT License – use, modify, cite freely.
