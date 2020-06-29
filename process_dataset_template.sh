#!/usr/bin/env bash
#The second to last parameter is number of data points to be processed from the original point (natsort order).
#If 0 is passed - process the entire dataset. If the number is negative take the n last entries from the dataset.

#The last parameter is a string of options divided by spaces:
#enable_gen_sh - sh files will be generated in the source directory prior to proceeding
#enable_overwrite - do not perform checks if cropped or resized folders exist - this will overwrite existing files
#disable_existence_check - do not perform checks if cropped or resized folders exist - this will ignore existing files and only generate new ones

bash ./process_dataset_exec.sh \
 /home/nedko/face_relight/dbs/data/stylegan_v0/test 1024 \
 /home/nedko/face_relight/dbs/stylegan_test 256 \
 /home/nedko/face_relight/dbs/data/stylegan_v0/normals \
 /home/nedko/face_relight/dbs/data/real/real_ffhq_256 \
 0 \
 "enable_gen_sh enable_overwrite"