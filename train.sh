python train_ser.py --dataroot /home/nedko/face_relight/dbs/stylegan_v0.3.1_256_crop/ --name model_256_lab_stylegan_0.3.1_10k_intensity_ambient_crop_shfix3 --input_mode LAB --gpu_ids 0,1 --model lightgrad59Ambient --direction AtoB  --checkpoints_dir ./outputs/ --display_winsize 128 --dataset_mode light3DULAB --display_id 0 --save_epoch_freq 1 --save_latest_freq 67625 --num_threads 10 --batch_size 20 --max_dataset_size 193380 --img_size 256 --n_synth 5 --n_first 5 --n_ids 10000 #--continue_train --epoch 8 #--enable_neutral