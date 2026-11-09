# Research Note — VisionLang AI

**Student:** Naran Deep  
**Project:** Evaluating Vision–Language Models for Image–Text Retrieval Using Contrastive Learning  
**Status:** Fill in only after running experiments. No results have been assumed.  

## Abstract (write 100–150 words)

Problem: [...]. Frozen pretrained CLIP models: [...]. Dataset/protocol: [...]. Both retrieval directions and metrics: [...]. Main observed result with measured value: [...]. Key limitation: [...].

## 1. Research question and hypothesis

**Question:** How do visual corruption and caption length affect zero-shot image–text retrieval performance in two pretrained CLIP encoders?

Pre-register simple hypotheses before inspecting the scores:
- **H1:** blur and grayscale degrade retrieval recall vs. clean images.
- **H2:** the two CLIP image encoders show different retrieval/robustness tradeoffs.
- **H3 (exploratory):** caption-length groups differ in retrieval performance; this is correlation, not causation.

## 2. Methods

**Dataset:** `intro/flickr8k` official `test` split (1,000 images, five captions each).  
**Pretrained models:** `openai/clip-vit-base-patch32`, `openai/clip-vit-base-patch16` (frozen, no finetuning).  
**Candidate sets:** 1,000 images for text→image; 5,000 captions for image→text.  
**Metrics:** R@1, R@5, R@10, MRR, median rank.  
**Uncertainty:** 95% image-group bootstrap CI conditioned on fixed gallery; seed 42.  
**Image transformations:** Gaussian blur radius 3, grayscale.  
**Caption bins:** ≤8 words; 9–13; ≥14. These bins are observational.  
**Compute environment:** [GPU/CPU, software versions], **timing:** [...].

## 3. Results — INSERT REAL OUTPUTS ONLY

| Model | Condition | Direction | R@1 | R@5 | R@10 | MRR |
|---|---|---|---:|---:|---:|---:|
| CLIP B/32 | clean | text→image | TBD | TBD | TBD | TBD |
| CLIP B/32 | clean | image→text | TBD | TBD | TBD | TBD |
| CLIP B/16 | clean | text→image | TBD | TBD | TBD | TBD |
| CLIP B/16 | clean | image→text | TBD | TBD | TBD | TBD |

**Caption-length findings:** [counts for each group and observed recall; discuss potential confounding].

**Robustness:** [measured absolute R@1 change in percentage points between clean and blurred/grayscale for each direction].

**Failure cases:** Provide 3–5 actual examples from `failures.csv` with your own visual inspection and comments. Do not rely on filenames alone to infer the image contents.

## 4. Interpretation and limits

- Contrastive vision–language representations make semantic ranking possible, but model ability depends on candidate-gallery size, data domain, and text/image complexity.
- This is a zero-shot evaluation of existing trained models, not a newly proposed model.
- Flickr8k images and reference captions have sampling/annotation biases.
- Comparing pretrained checkpoints cannot attribute performance differences solely to patch size; architectures, training details and hyperparameters can differ.
- Caption length is a subgroup correlate, not a manipulated causal intervention.
- Confidence intervals are conditional on this fixed retrieval gallery and its image-group sampling.

## 5. Conclusion and next study

[Summarize actual findings and propose controlled caption perturbations, a second dataset (e.g., Flickr30k), or a properly isolated validation/test fine-tuning study. Do not claim future results.]

## 6. References

1. Radford et al. 2021. https://arxiv.org/abs/2103.00020
2. Hodosh et al. 2013. https://www.jair.org/index.php/jair/article/view/10833
3. Flickr8k data card. https://huggingface.co/datasets/intro/flickr8k
