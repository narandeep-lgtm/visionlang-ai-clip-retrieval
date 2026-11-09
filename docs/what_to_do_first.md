# Start here: from ZIP to real EPFL evidence

1. Download the repository ZIP and `notebooks/colab_quickstart.ipynb` (also provided separately).
2. Open https://colab.research.google.com/ and upload the notebook.
3. Set Runtime → Change runtime type → T4 GPU (if available).
4. Run the notebook from the top; in the upload cell choose the ZIP. The project is extracted automatically.
5. First, run the 100-image verification and check for errors. Then run the full 1000-image test split with both models.
6. Download the **actual experiment output** from Colab. The generated CSV and PNG files should come from this run, not from fabricated metrics.
7. Use `docs/research_report_template.md` to write what the numbers mean. Inspect the failure cases manually and write your own observations.
8. Publish the code and report to GitHub. Only then add the link to your CV, clearly labeling your work as a reproducible *evaluation* of pretrained models.

**Do not claim "trained CLIP", "invented a novel contrastive method", "published research", or high model accuracy without supporting evidence.**

Suggested truthful CV entry **after completion**:

**VisionLang AI — Zero-shot Vision–Language Retrieval Benchmark** | Python, PyTorch, CLIP, Flickr8k
- Implemented reproducible bidirectional image–text retrieval evaluation of pretrained CLIP ViT-B/32 and ViT-B/16, reporting R@1/5/10 and image-group bootstrap uncertainty.
- Analyzed retrieval errors, caption-length subgroups, and robustness to blur/grayscale; documented findings in a technical report with code and experimental outputs.

This is a **template**, not a statement that the experiments have already been completed.
