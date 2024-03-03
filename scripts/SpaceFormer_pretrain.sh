device=0
cell_mask_rate=0.5
gene_mask_rate=0.3
gamma=0.2
ffn=3999
dropout=0.1
weightdecay=0.0001
input=3999
for seed in 0 1 2 3 4
do
    python ../pretrain.py --train_ids 1 2 3 4 5 6 7 --test_ids 0 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 2 3 4 5 6 7 --test_ids 1 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 3 4 5 6 7 --test_ids 2 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 2 4 5 6 7 --test_ids 3 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 2 3 5 6 7 --test_ids 4 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 2 3 4 6 7 --test_ids 5 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 2 3 4 5 7 --test_ids 6 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
    python ../pretrain.py --train_ids 0 1 2 3 4 5 6 --test_ids 7 --model SpaceFormer --weight_decay $weightdecay --seed $seed --device $device --ffn_dim $ffn --gamma $gamma --spatial --dropout $dropout --gene_mask_rate $gene_mask_rate --cell_mask_rate $cell_mask_rate --loss_fn sce --input_dim $input
done