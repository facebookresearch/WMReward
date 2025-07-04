# srun --gpus-per-node=8 --partition=learn --time=08:00:00 --cpus-per-task 48 --pty /bin/bash -l

srun --gpus-per-node=1 --partition=learn --time=08:00:00 --cpus-per-task 48 --qos=dev --pty /bin/bash -l