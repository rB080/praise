# -------------------------- tiny usage demo --------------------------
if __name__ == "__main__":
    import os
    from PIL import Image
    from wrapped_qwen import QwenVLClipish

    wrapper = QwenVLClipish(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        device_map="cuda:3",
        dtype="bfloat16",
    )

    # Encode two images + two texts
    img_paths = [
        "/export/livia/home/vision/Rbhattacharya/work/data/data/market1501/Market-1501-v15.09.15/bounding_box_test/0004_c1s6_016996_02.jpg",
        "/export/livia/home/vision/Rbhattacharya/work/data/data/market1501/Market-1501-v15.09.15/bounding_box_test/0004_c1s6_017046_01.jpg",
    ]
    images = [Image.open(p).convert("RGB") for p in img_paths if os.path.exists(p)]
    # breakpoint()
    img_emb = wrapper.encode_image(images)            # (B, D)
    txt_emb = wrapper.encode_text(["a person wearing a red shirt", 
                                    "a person wearing a blue shirt", 
                                    "a person wearing a green shirt", 
                                    "a person wearing a yellow shirt", 
                                    "The individual is wearing a bright red t-shirt with a white logo on the left chest area, and a pair of blue shorts. The shorts are likely made of a cotton blend, and the t-shirt appears to be a standard fit. The person has a medium to dark skin tone, and their hair is not visible as it is obscured by the t-shirt. They are carrying a black backpack on their right shoulder, and their footwear consists of white sneakers. The person's gait suggests they are in motion, with one leg slightly bent, indicating a walking posture. The background is indistinguishable due to the focus on the person, but it appears to be an outdoor setting with a possible sidewalk or path."
                                    ])  # (2, D)
    breakpoint()
    sims = wrapper.similarity(img_emb, txt_emb)       # (B, 2)
    top_idx, top_scores = wrapper.topk(img_emb, txt_emb, k=1)
    print("Top-1 text per image:", top_idx.squeeze(-1).tolist(), top_scores.squeeze(-1).tolist())

    # Optional: generate
    out = wrapper.generate(images[:1], "Describe only the clothing briefly.")
    print(out[0])