CUDA_VISIBLE_DEVICES=3 python ./utils/miner.py \
  --image_root /export/livia/home/vision/Rbhattacharya/work/data/data/market1501/Market-1501-v15.09.15/bounding_box_train \
  --dataset market1501 \
  --out_dir /export/livia/home/vision/Rbhattacharya/work/tdop/attribute_banks/market \
  --sample_per_pid 4 \
  --max_images 3000 \
  --device cuda:0 \
  --dtype bfloat16 \
  --min_count 5