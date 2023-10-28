#code for feature extraction and readout

model_save_name="mae"
model_name="physion_model.MAE_huge"
gpu=1
save_dir="./test_feats"
mode="seg"
data_root_path="/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/"
#cmd="physion_feature_extract --model_save_name $model_save_name --model $model_name --gpu $gpu --dir_for_saving $save_dir --mode $mode --data_root_path $data_root_path --model_path /ccn2/u/rmvenkat/code/deploy_code/mae/mae_visualize_vit_large.pth"
#echo $cmd
#eval "$cmd"

save_dir_model="$save_dir/$model_save_name/$mode"


if [ "$mode" = "seg" ]; then
    cmd="physion_train_seg_readout --train_features_hdf5 $save_dir_model/train_features.hdf5  --test_features_hdf5 $save_dir_model/test_features.hdf5  --model_type $model_save_name --gpu_id $gpu"
else
    cmd="physion_train_readout --data-path $save_dir_model/train_features.hdf5 --data-type $model_save_name'_'$mode  --test-path $save_dir_model/test_features.hdf5 --scenario-name observed --train-scenario-indices $save_dir_model/train_json.json --test-scenario-indices $save_dir_model/test_json.json  --scenario features --test-scenario-map $save_dir_model/test_scenario_map.json --one-scenario all --ocp --save_path $save_dir_model"
fi

echo $cmd
eval "$cmd"

