run_in_tmux_session() {
    # Parameters
    local save_dir="$1"
    local model_save_name="$2"
    local model_name="$3"
    local weights="$4"
    local gpu="$5"
    local data_root_path="$6"
    local env_path="$7"
    modes="ocp ocd_focussed seg"

    for mode in $modes
    do
        local dir="$save_dir/$model_save_name/$mode"
        local cmd="physion_feature_extract --model_save_name $model_save_name --model_path $weights --model $model_name --gpu $gpu --dir_for_saving $save_dir --mode $mode --data_root_path $data_root_path"

        session_name=$model_save_name'_'$mode
        local cmd_tmux="tmux kill-session -t '$session_name'"
        echo $cmd_tmux
        eval "$cmd_tmux"

        # Create new tmux session
        cmd_tmux="tmux new -d -s '$session_name'"
        echo $cmd_tmux
        eval "$cmd_tmux"

        # Activate the specific conda environment
        local cmd_conda="tmux send-keys -t '$session_name' \"conda activate $env_path\" C-m"
        echo $cmd_conda
        eval "$cmd_conda"

        #command to change ulimit -n
        local cmd_ulimit="tmux send-keys -t '$session_name' \"ulimit -n 65535\" C-m"
        echo $cmd_ulimit
        eval "$cmd_ulimit"

        local cmd_save_features="tmux send-keys -t '$session_name' \"$cmd\" C-m"
        echo $cmd_save_features
        eval "$cmd_save_features"

    done

}

#regenerate_from_old_commit
save_dir="/ccn2/u/rmvenkat/data/final_set_cvpr_results_fixed_bugs/"
data_root_path="/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/"
env_path="/ccn2/u/rmvenkat/across_conda/envs/vmae_flash_2"

model_save_name="mae_45m_large"
model_name="physion_model.MAE_large_256"
gpu=0
weights="/ccn2/u/rmvenkat/code/deploy_code/mae/mae_45_wts/large.pth"
run_in_tmux_session $save_dir $model_save_name $model_name $weights $gpu $data_root_path $env_path

model_save_name="mae_45m_base"
model_name="physion_model.MAE_base"
gpu=1
weights="/ccn2/u/rmvenkat/code/deploy_code/mae/mae_45_wts/base.pth"
run_in_tmux_session $save_dir $model_save_name $model_name $weights $gpu $data_root_path $env_path

model_save_name="mae_base"
model_name="physion_model.MAE_base"
gpu=2
weights="/ccn2/u/rmvenkat/code/deploy_code/mae/mae_weights/mae_pretrain_vit_base.pth"
run_in_tmux_session $save_dir $model_save_name $model_name $weights $gpu $data_root_path $env_path

model_save_name="mae_large"
model_name="physion_model.MAE_large"
gpu=3
weights="/ccn2/u/rmvenkat/code/deploy_code/mae/mae_weights/mae_pretrain_vit_large.pth"
run_in_tmux_session $save_dir $model_save_name $model_name $weights $gpu $data_root_path $env_path

model_save_name="mae_huge"
model_name="physion_model.MAE_huge"
gpu=4
weights="/ccn2/u/rmvenkat/code/deploy_code/mae/mae_weights/mae_pretrain_vit_huge.pth"
run_in_tmux_session $save_dir $model_save_name $model_name $weights $gpu $data_root_path $env_path