#!/bin/bash
# Hebbian CTM with Oja's rule training on mazes-small dataset for Mac (Apple Silicon)
# Oja's rule is a normalized Hebbian learning rule that prevents unbounded weight growth
# ΔW_ij = η * act_i * (act_j - W_ij * act_i)

python -m tasks.mazes.train \
--dataset mazes-small \
--maze_route_length 50 \
--cirriculum_lookahead 5 \
--model ctm \
--d_model 1024 \
--d_input 256 \
--backbone_type resnet18-1 \
--synapse_depth 8 \
--heads 4 \
--n_synch_out 128 \
--n_synch_action 128 \
--neuron_select_type random-pairing \
--memory_length 25 \
--iterations 50 \
--training_iterations 100001 \
--lr 1e-4 \
--batch_size 64 \
--batch_size_test 32 \
--n_test_batches 50 \
--log_dir logs/hebbian-oja/mazes-small-mac \
--track_every 2000 \
--save_every 1000 \
--keep_checkpoint_history \
--use_hebbian \
--use_oja \
--hebbian_lr 0.01 \
--hebbian_decay 0.999 \
--lateral_strength 0.1 \
--lateral_injection pre_synapse
