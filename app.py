"""Optional Gradio demo using embeddings CREATED by a completed benchmark run.

Launch only after: python -m visionlang --models clip-b32 --max-images 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from visionlang.models import CLIPEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="results/index_clip-b32.npz")
    parser.add_argument("--model", choices=["clip-b32", "clip-b16"], default="clip-b32")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not Path(args.index).is_file():
        raise FileNotFoundError("Run `python -m visionlang --models clip-b32 --max-images 0` first to generate embeddings")
    import gradio as gr
    from datasets import load_dataset
    index = np.load(args.index, allow_pickle=False)
    filenames = [str(x) for x in index["filenames"]]
    image_vecs = index["image_embeddings"]
    text_vecs = index["caption_embeddings"]
    captions = [str(x) for x in index["captions"]]
    encoder = CLIPEncoder(args.model)
    if f"index_{args.model}.npz" != Path(args.index).name:
        raise ValueError("--model must match the model stored in the index filename")
    print("Loading test images for demonstration (retrieval evaluation remains separate)...")
    ds = load_dataset("intro/flickr8k", split="test")
    lookup = {str(row["file_name"]): row["image"].convert("RGB") for row in ds}
    images = [lookup[name] for name in filenames]

    def text_to_image(query, k):
        if not query.strip():
            return []
        q = encoder.texts([query.strip()])[0]
        scores = image_vecs @ q
        ids = np.argsort(-scores)[:int(k)]
        return [(images[int(i)], f'#{rank} · cosine {scores[i]:.3f} · {filenames[int(i)]}')
                for rank, i in enumerate(ids, 1)]

    def image_to_text(im, k):
        if im is None:
            return "Upload an image to see matching descriptions."
        q = encoder.images([im.convert("RGB")])[0]
        scores = text_vecs @ q
        ids = np.argsort(-scores)[:int(k)]
        return "\n".join(f'{rank}. {captions[int(i)]}  (cosine: {scores[i]:.3f})'
                         for rank, i in enumerate(ids, 1))

    with gr.Blocks(title="VisionLang: Research Demo") as demo:
        gr.Markdown("# VisionLang AI — Cross-modal Retrieval\nFrozen CLIP embeddings. Scores are similarity, **not probability**. This is a demo, not an evaluation metric.")
        k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K")
        with gr.Tab("Text → Images"):
            query = gr.Textbox(label="Describe an image", value="a child running with a dog")
            search_btn = gr.Button("Retrieve images")
            gallery = gr.Gallery(label="Best matching test images", columns=3, height=350)
            search_btn.click(text_to_image, inputs=[query, k], outputs=gallery)
        with gr.Tab("Image → Text"):
            img = gr.Image(type="pil", label="Upload image")
            search_img = gr.Button("Retrieve captions")
            descriptions = gr.Textbox(label="Closest reference captions", lines=8)
            search_img.click(image_to_text, inputs=[img, k], outputs=descriptions)
        gr.Markdown("Dataset: Flickr8k test split. Original images remain with their source owners; review dataset usage terms before publishing screenshots.")
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
