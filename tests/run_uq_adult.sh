#!/bin/bash

echo "Starting Final Adult Dataset Experiment Suite"
echo "This will run all methods over multiple seeds."
echo "-------------------------------------------------"

# === Configuration ===
SEEDS=(6 12 13 523 972394)
DATA_ROOT="./data"
MODEL_NAME="MLPTabular"
BENCHMARK="Adult"
RESULTS_ROOT="./results/${BENCHMARK}"

# Ensure results directory exists
mkdir -p $RESULTS_ROOT

COMMON_ARGS="--benchmark ${BENCHMARK} --model ${MODEL_NAME} --data_root ${DATA_ROOT} --batch_size 256"

# Specific arguments for the new methods from your friends' scripts
SUBSPACE_ARGS="--method subspace --subspace_dim 20 --subspace_method random --eig_steps 100"
SWAG_LAPLACE_ARGS="--method swag_laplace --n_samples 30 --subspace_dim 20 --subspace_method random --eig_steps 100"


# Loop over each seed
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "       RUNNING FOR SEED: $seed"
    echo "========================================"

    # Use a seed for the model training/initialization
    SEED_ARGS="--model_seed ${seed}"

    # === 1. BASELINE EXPERIMENTS (Standard Train/Test Split) ===
    echo "--> RUNNING BASELINE: MAP"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method map --run_name "${BENCHMARK}/map_baseline_seed${seed}"

    echo "--> RUNNING BASELINE: LA (Last-Layer)"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method laplace --subset_of_weights last_layer --run_name "${BENCHMARK}/laplace_ll_baseline_seed${seed}"

    echo "--> RUNNING BASELINE: LA* (Last-Layer, Full EF)"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method laplace --subset_of_weights last_layer --hessian_structure full --approx_type ef --run_name "${BENCHMARK}/laplace_star_baseline_seed${seed}"

    echo "--> RUNNING BASELINE: Subspace Laplace"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} ${SUBSPACE_ARGS} --run_name "${BENCHMARK}/subspace_baseline_seed${seed}"

    echo "--> RUNNING BASELINE: SWAG-Laplace"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} ${SWAG_LAPLACE_ARGS} --run_name "${BENCHMARK}/swag_laplace_baseline_seed${seed}"


    # === 2. DOMAIN SHIFT EXPERIMENTS (Using standard Last-Layer LA) ===
    echo "--> RUNNING DOMAIN SHIFT: Male-to-Female"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method laplace --subset_of_weights last_layer --domain_shift_gender male_to_female --run_name "${BENCHMARK}/shift_male_to_female_seed${seed}"

    echo "--> RUNNING DOMAIN SHIFT: Female-to-Male"
    python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method laplace --subset_of_weights last_layer --domain_shift_gender female_to_male --run_name "${BENCHMARK}/shift_female_to_male_seed${seed}"


    # === 3. NOISE INTENSITY EXPERIMENTS (Using standard Last-Layer LA) ===
    for intensity in 0.1 0.25 0.5 0.75 1.0; do
        echo "--> RUNNING NOISE INTENSITY: ${intensity}"
        python tests/uq.py ${COMMON_ARGS} ${SEED_ARGS} --method laplace --subset_of_weights last_layer --noise_intensity ${intensity} --run_name "${BENCHMARK}/noise_${intensity}_seed${seed}"
    done
done

echo ""
echo "---------------------------------------"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results are saved in ${RESULTS_ROOT}"
