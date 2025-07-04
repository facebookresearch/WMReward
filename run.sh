# CUDA_VISIBLE_DEVICES=7 nohup python3 generate_cosmos_guidance_i2v.py > log.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python3 generate_guidance.py > log.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=7 python3 generate_cosmos_guidance_i2v.py

CUDA_VISIBLE_DEVICES=7 nohup python reproduce_intphy_v2.py --data_path /home/yjianhao/project/video_guidance/dev/O1 > log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python test_vjepa_loss_pipeline.py > log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python generate_cosmos_i2v.py > log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python generate.py > log1.txt 2>&1 &
# check number files
find . -type f | awk -F/ 'NF>2{print $2}' | sort | uniq -c