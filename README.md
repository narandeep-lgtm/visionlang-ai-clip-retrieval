# VisionLang AI

### Evaluating Vision–Language Models for Image–Text Retrieval Using Contrastive Learning

**Author:** Naran Deep · BS Artificial Intelligence · independent learning / research-style reproducibility project

> **Status:** benchmark implementation and offline unit tests provided. **No benchmark results are claimed until the pretrained models are executed on the specified Flickr8k test split.** This repository uses frozen, pretrained contrastive models; it does **not** claim new contrastive pretraining or independent invention of CLIP.

[Original CLIP paper](https://arxiv.org/abs/2103.00020) · [Hugging Face CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip) · [Flickr8k dataset card](https://huggingface.co/datasets/intro/flickr8k)

## Research question

**How do visual corruption and caption length affect zero-shot image–text retrieval performance in two pretrained CLIP encoders?**

The study compares `openai/clip-vit-base-patch32` and `openai/clip-vit-base-patch16` on the **same full Flickr8k test gallery**. It reports image→text and text→image retrieval separately and examines failures rather than presenting only a web demo. Models were pretrained elsewhere; no training occurs in this repository.

## Setup (Google Colab with GPU recommended)

Python 3.10–3.12 recommended; a Colab notebook is provided at `notebooks/colab_quickstart.ipynb`.

```bash
pip install -r requirements.txt
pytest -q
```

**First dataset download:** Hugging Face `intro/flickr8k` is about 1.11 GB. Pretrained weights also require internet. Nothing is bundled or redistributed in this repository.

### A. Quick functional demonstration (NOT a benchmark)

```bash
python -m visionlang --models clip-b32 --max-images 100 --bootstrap 100 --corruptions grayscale --outdir results/quick
```

This runs on 100 reproducibly sampled test images with a much easier candidate gallery; don't report those scores as comparable to the 1,000-image standard benchmark.

### B. Full experiment (use for a research report)

```bash
python -m visionlang --models clip-b32 clip-b16 --max-images 0 --seed 42 --bootstrap 500 --corruptions grayscale blur --outdir results/full
```

This runs on the **entire official 1,000-image test set (5 captions per image)**. On a Colab T4 GPU, set `--batch-size 8` if memory is limited. CPU execution can be slow. Outputs are calculated from actual embeddings, never hard-coded.

### C. Optional interactive demo

```bash
pip install -r requirements-demo.txt
python app.py --index results/full/index_clip-b32.npz --model clip-b32
```

**What demo does:** real text→image search, plus uploaded image→matching-caption search using cosine similarity. This is not a substitute for quantitative evaluation. The demo loads the Flickr8k test images independently; do not publish those images without reviewing their usage permissions.

## Methods

```
Flickr8k official test (1000 images, 5 captions/image)
          |                        |
  frozen CLIP image encoder    frozen CLIP text encoder
          |                        |
    unit embeddings          unit embeddings
          +-------- cosine --------+
                    |
      rank against fixed candidate galleries
                    |
   R@1 / R@5 / R@10, MRR, median rank
                    |
      bootstrap CIs and failure analysis
                    |
  model comparison / caption bins / corruption
```

- **Text → image:** each caption is a query, its source image is relevant, and there are 1,000 image candidates.
- **Image → text:** each image is a query, any of its *five* human captions is relevant, and there are 5,000 caption candidates. A top-1 result is a success if **any** paired caption ranks first.
- **Similarity:** cosine similarity of L2-normalized projected embeddings, not calibrated probabilities.
- **95% bootstrap confidence intervals:** resample image groups, not sibling captions independently; conditional on the fixed gallery and selected dataset. They do not capture all sources of uncertainty.
- **Caption-length analysis:** predefined `<=8`, `9–13`, `>=14` word bins (text→image). This is an *observational subgroup comparison*: it does not isolate caption length as a causal effect. Image/caption content is a possible confounder.
- **Image corruption:** Gaussian blur radius 3 and grayscale applied to the same test images, with captions unchanged. Compare condition-specific retrieval recall to clean images.
- **No leakage:** Flickr8k officially separated test images only. No test captions enter any fitting or hyperparameter selection stage. No finetuning or model training is claimed.

## Saved artifacts

| Output | Interpretation |
|---|---|
| `results/full/metrics.csv` | R@1/R@5/R@10, MRR, median ranks for both directions |
| `results/full/confidence_intervals.json` | Fixed-gallery image-group bootstrap intervals |
| `results/full/caption_length.csv` | Performance across prespecified caption-length bins |
| `results/full/failures.csv` | Top-ranked mismatches and corresponding image filenames |
| `results/full/recall_curves.png` | Comparative clean-test Recall@K figure |
| `results/full/run_metadata.json` | Dataset split, candidate counts, seed, software version, runtime |
| `results/full/index_clip-b32.npz` | Embeddings for optional local demo |

Do not add unmeasured numbers to your CV or report. Use the full split before making comparison claims. Consider including measured GPU type and time, plus a table of ten qualitative failures and what each teaches.

## Evaluation checklist

- [ ] Run `pytest -q` successfully
- [ ] Run 100-image smoke test and inspect all outputs
- [ ] Run full 1,000-image test split for **both** models
- [ ] Check both directions and inspect surprising rank outcomes
- [ ] Read 10 failure examples and document patterns
- [ ] Produce a table of actual metrics with proper split/candidate counts
- [ ] Document limitations, dataset biases, possible confounds
- [ ] Write your own 1–2-page report using `docs/research_report_template.md`
- [ ] Be able to explain cosine similarity, Recall@K, CLIP and the methodology unaided

## Project limitations and scientific integrity

1. This is **evaluation**, not novel model architecture or contrastive pretraining. To claim a method improvement, actually implement and measure it against the same baselines.
2. Flickr8k image rights remain with original photographers; dataset-card license metadata should not be interpreted as blanket permission to redistribute original photos.
3. Scores for a 100-image quick run cannot be compared to full-gallery scores; gallery size strongly affects recall.
4. Accuracy is not a property of the overall project: measured **retrieval Recall@K** is the relevant primary metric.
5. Descriptive caption-length bins do not prove a causal relationship; biases and semantic complexity may differ.
6. Model outputs are imperfect, may encode social biases, and cosine scores are not probabilities.
7. Code has unit tests for mathematical/retrieval behavior. End-to-end pretrained model and dataset execution needs internet/weights, and must be verified where it runs.

## How to discuss it truthfully in an EPFL application

> Independently studied and evaluated pretrained vision–language retrieval using CLIP, implementing bidirectional Recall@K analysis, image-corruption robustness experiments, caption-length subgroup analysis, reproducibility controls, and qualitative error inspection. Used AI coding assistance, then validated the implementation and documented results.

**Only use this wording once you have actually run and examined the experiments.** Cite specific verified metrics only after completing the full benchmark. Clearly disclose AI assistance if asked.

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. https://arxiv.org/abs/2103.00020
- Hodosh, Young & Hockenmaier (2013). *Framing Image Description as a Ranking Task*. https://www.jair.org/index.php/jair/article/view/10833
- Hugging Face `intro/flickr8k` dataset: https://huggingface.co/datasets/intro/flickr8k

Code license: MIT. Trained weights and Flickr8k photographs are **not** covered by this repository's code license.
