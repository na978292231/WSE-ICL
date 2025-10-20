#export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=/data/llmweights/${LLM}
DATA_DIR=./data/




DATASET=wiki-zh
for  N_DEMO_SHOT in 0 1 4 8 16 32 64 128; do
     for N_TRAIN_SHOT in 4 8 16 32 64 128; do
          if ((N_DEMO_SHOT < N_TRAIN_SHOT))
          then
          for SEED in 1 2 3 4 5; do
          python3 fads-icl.py \
              --llm_dir ${LLM_DIR} \
              --dataset ${DATASET} \
              --data_dir ${DATA_DIR} \
              --n_train_shot ${N_TRAIN_SHOT} \
              --n_demo_shot ${N_DEMO_SHOT} \
              --seed ${SEED} \
              --output_dir ./output/fads-icl/${LLM} \
              --feature_choose all
          done
         fi
     done

 done
