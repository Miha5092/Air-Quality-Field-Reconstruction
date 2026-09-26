#!/bin/bash

python -m experiments.real_dataset_evaluation --experiment_name temporal --model_types vitae --epochs 0 --noise none --pollutant_type all --timesteps 6 --model_path paper_results/trained_models/vitae/t6.pth
python -m experiments.real_dataset_evaluation --experiment_name temporal --model_types vunet --epochs 0 --noise none --pollutant_type all --timesteps 4 --model_path paper_results/trained_models/vunet/t4_42.pth
python -m experiments.real_dataset_evaluation --experiment_name temporal --model_types clstm --epochs 0 --noise none --pollutant_type all --timesteps 8 --model_path paper_results/trained_models/unknown_model/t8_42.pth

python -m experiments.real_dataset_evaluation --experiment_name temporal --model_types diffusion --epochs 0 --noise none --pollutant_type all --timesteps 1 --model_path paper_results/trained_models/diffusion/all/no_noise.pth
