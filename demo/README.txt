#Instructions for running the Web demo

=== Prerequisites ===

- Ubuntu
- Miniconda
- CUDA 10.0

=== Installation ===

1. Create the conda environment with all the dependencies required:

conda env create -f environment.yml

=== Deployment ===

1. Run Celery. If ran with 1 worker (-c 1), cpu will be used, otherwise - GPU:

celery -A dnr_demo worker -c 1 --loglevel=info

2. Run the server:

python3 manage.py dnr_demo