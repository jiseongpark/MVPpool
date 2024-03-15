import argparse
from mvp_train import Trainer


def parse_args():
    
    parser = argparse.ArgumentParser(description='Multi-View Pruning Model Training')
    
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--GNN_type', type=str, default='GCS',
                       choices=['GCN', 'GCS', 'GIN', 'ARMA'])
    parser.add_argument('--n_channels', type=int, default=32)
    parser.add_argument('--activ', type=str, default='relu')
    parser.add_argument('--mincut_H', type=int, default=16)
    parser.add_argument('--GNN_l2', type=float, default=1e-4)
    parser.add_argument('--pool_l2', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--es_patience', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--dataset', type=str, default='PROTEIN/',
                       choices=['PROTEIN', 'IMDB-BINARY', 'DD', 'COLLAB', 'NCI', 'IMDB-MULTI'])
    parser.add_argument('--history_path', type=str, default='./history/GMT_test')
    parser.add_argument('--method', type=str, default='mincut_pool',
                       choices=['flat', 'dense', 'diff_pool', 'top_k_pool', 'mincut_pool', 'sag_pool'])
    parser.add_argument('--anomaly_bias', type=float, default=0.6)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--temperature', type=list, default=[1.0, 1.0])
    parser.add_argument('--K', type=list, default=[20, 10])
    parser.add_argument('--gnn_depth', type=int, default=1)
    parser.add_argument('--viewpoints', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--view_gen', type=str, default='R',
                       choices=['R', 'S', 'E', 'N'])

    return parser.parse_known_args()


if __name__ == '__main__':
    
    args, unparsed = parse_args()
    
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
        
    trainer = Trainer(args)
    trainer.train('pre')
    trainer.train('post_ox')