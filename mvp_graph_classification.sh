python mvp_main.py --viewpoints 4 \
                   --hidden_dim 8  \
                   --embed_dim 32  \
                   --view_gen R  \
                   --GNN_type GCS  \
                   --es_patience 6  \
                   --learning_rate 0.0005  \
                   --dataset PROTEIN  \
                   --history_path ./history/$1  \
                   --method mincut_pool  \
                   --runs 1

