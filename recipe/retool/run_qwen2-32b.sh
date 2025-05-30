set -x

# ================= data/model/tool =================
DATA_ROOT=

dapo_math_17k=${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024
aime_2025=${DATA_ROOT}/dataset/yentinglin/aime_2025
model_path=${DATA_ROOT}/JoeYing/ReTool-Qwen-32B-SFT

train_files="['$dapo_math_17k']"
test_files="['$aime_2024', '$aime_2025']"

# sandbox
chat_scheduler=recipe.retool.chat_scheduler.ToolChatCompletionScheduler
sandbox_fusion_url="https://***.apigateway-cn-beijing.volceapi.com/run_code"

# wandb
project_name=wuxibin_retool
experiment_name=qwen2-32b_gae
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95

critic_warmup=10

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=4 # train

train_batch_size=1024
ppo_mini_batch_size=256
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
critic_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4 ))


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=examples/ppo_trainer/runtime_env.yaml \
    --entrypoint-num-cpus=1 \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/chat_scheduler.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/chat_scheduler.py \
    custom_reward_function.name=compute_score \
    reward_model.sandbox_fusion.url=$sandbox_fusion_url \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    critic.optim.lr=$critic_lr \
    critic.model.use_remove_padding=True \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.model.fsdp_config.param_offload=$offload \
    critic.model.fsdp_config.optimizer_offload=$offload \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name-$(date +"%Y-%m-%d-%H-%M-%S") \
    trainer.n_gpus_per_node=$ARNOLD_WORKER_GPU \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=10 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
