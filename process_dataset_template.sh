#!/usr/bin/env bash
#The last parameter is a string of options divided by spaces:
#enable_gen_sh - sh files will be generated in the source directory prior to proceeding
#enable_overwrite - do not perform checks if cropped or resized folders exist - this will overwrite existing files
#disable_existence_check - do not perform checks if cropped or resized folders exist - this will ignore existing files and only generate new ones

bash ./process_dataset_exec.sh \
 /home/nedko/face_relight/dbs/data/stylegan_v0/test 1024 \
 /home/nedko/face_relight/dbs/stylegan_test 256 \
 /home/tushar/data2/rendering_pipeline/stylegan_final_30k \
 /home/nedko/face_relight/dbs/data/real/real_ffhq_256 \
 "enable_gen_sh"