set -ex

streamlit run visualize.py \
    --data_dir "datasets/mini_coco_2014/Images/" \
    --model_name_or_path "llava-hf/llava-1.5-7b-hf" \
    