# Instructions for running the Web demo

### Prerequisites

- Ubuntu
- Miniconda
- CUDA 10.0

### Installation 

1. Create the conda environment with all the dependencies required:

        conda env create -f environment.yml
        conda activate fr
        cd dnr_demo
        pip install -r requirements.txt

### Deployment 

1. Run Celery. If ran with 1 worker (-c 1), cpu will be used, otherwise - GPU:

        celery -A dnr_demo worker --loglevel=info

2. Run the server:

        python manage.py runserver 0.0.0.0:8070