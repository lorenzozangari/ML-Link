from utils.params import set_params
from utils.optimization import EarlyStopping
import torch
import dgl
import time
import torch.optim as optim
from dgl.sampling import global_uniform_negative_sampling
from models.main_m import Mm
from input_data.load import load_data
from sklearn.metrics import roc_auc_score, average_precision_score
from models.link_predictor import MLPPredictor, LinkPredictor
from utils.d_util import print_arguments, write_results
from tqdm.auto import tqdm
from utils.logger import Logger
import torch.nn.functional as F
import os
from utils.util import init_seed
import utils.const as C


def build_model(args, n_layers, input_dim):
    no_struct = args.no_struct
    no_gnn = args.no_gnn
    model = Mm(n_layers=n_layers, dropout=args.dropout, no_struct=no_struct, no_gnn=no_gnn, psi=args.psi,
               edge_dim=args.edge_dim, node_dim=args.node_dim, phi_dim=args.phi_dim,
               input_dim=input_dim, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden,
               heads=args.n_heads, attn_dropout=args.attn_dropout, residual=True, aggregation=args.heads_mode,
               activation=F.elu,  eps=1e-8, f_dropout=0.7)

    return model


def get_predictor(op='dot',
                  args=None, dim=None):
    if op.lower() == 'mlp':
        return MLPPredictor(dim, dropout=args.dropout)
    return LinkPredictor(op)


def generate_negative_samples(g, n_l, n):
    g_train_negs = []
    for l_id in range(n_l):
        n_edges = g[l_id].number_of_edges()
        neg_g = dgl.graph(global_uniform_negative_sampling(g[l_id],
                          num_samples=n_edges, exclude_self_loops=True, replace=False), num_nodes=n)
        g_train_negs.append(neg_g)
    return g_train_negs


def compute_loss(pos_score, neg_score, device, eps=1e-8):
    pos_loss = -torch.log(pos_score + eps).mean()
    neg_loss = -torch.log(1 - neg_score + eps).mean()
    return pos_loss + neg_loss


def compute_score(pos_scores, neg_scores, scores=None):
    results = {}

    t_scores = torch.cat([pos_scores, neg_scores]).numpy()
    t_labels = torch.cat(
         [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]).numpy()

    if scores is None:
        auc_test = roc_auc_score(t_labels, t_scores)
        results['auc'] = auc_test
        ap_test = average_precision_score(t_labels, t_scores)
        results['ap'] = ap_test
    else:
        if 'auc' in scores:
            auc_test = roc_auc_score(t_labels, t_scores)
            results['auc'] = auc_test
        if 'ap' in scores:
            ap_test = average_precision_score(t_labels, t_scores)
            results['ap'] = ap_test
    return results


def eval(g_supra, g_train, g_test_pos, g_test_neg, g_val_pos, g_val_neg,
         p, feats, inter, model, predictor,
         return_scores=False):
    results = {}
    model.eval()
    if predictor is not None:
        for lid in range(len(predictor)):
            predictor[lid].eval()

    with torch.no_grad():
        pos_score_test, _, _ = model(g_supra, g_train, p, g_test_pos, feats, predictor, inter)
        neg_score_test, _, _ = model(g_supra, g_train, p, g_test_neg, feats, predictor, inter)
        pos_score_test = torch.vstack(pos_score_test).detach().cpu()
        neg_score_test = torch.vstack(neg_score_test).detach().cpu()
        if return_scores:
            return pos_score_test, neg_score_test
        t_scores = torch.cat([pos_score_test, neg_score_test]).numpy().squeeze(-1)
        t_labels = torch.cat(
            [torch.ones(pos_score_test.shape[0]), torch.zeros(neg_score_test.shape[0])]).numpy()

        pos_score_val, _, _ = model(g_supra, g_train, p, g_val_pos, feats, predictor, inter)
        neg_score_val, _, _ = model(g_supra, g_train, p, g_val_neg, feats, predictor, inter)
        pos_score_val = torch.vstack(pos_score_val).detach().cpu()
        neg_score_val = torch.vstack(neg_score_val).detach().cpu()

        v_scores = torch.cat([pos_score_val, neg_score_val]).numpy().squeeze(-1)
        v_labels = torch.cat(
            [torch.ones(pos_score_val.shape[0]), torch.zeros(neg_score_val.shape[0])]).numpy()

        auc_test = roc_auc_score(t_labels, t_scores)
        auc_val = roc_auc_score(v_labels, v_scores)
        ap_test = average_precision_score(t_labels, t_scores)
        ap_val = average_precision_score(v_labels, v_scores)

        results['auc'] = (auc_val, auc_test)
        results['ap'] = (ap_val, ap_test)

    return results


def train(config):
    dataset = config.dataset
    device = config.device
    seed = int(config.seed)
    omn = config.omn.strip().lower()  # Across-layer contexts
    omn = None if (omn == 'none' or config.psi == .0 or config.no_struct) else omn or None
    if omn is not None:
        omn = omn.split(';')
        omn.sort()
        assert all((omni == C.MAAN or omni == C.OAN) for omni in omn)
        print(f'OMN : {omn}')
    else:
        print('OMN set is empty!')

    lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0

    loggers = {
        'ap': Logger(config.runs, config),
        'auc': Logger(config.runs, config)
    }
    all_pos_scores = []
    all_neg_scores = []
    epoch_val = 5

    for run in range(config.runs):
        init_seed(run)
        print(f'Run : {run + 1}')
        g_supra, g_train, g_train_pos, g_test_pos, g_test_neg, g_val_pos, g_val_neg, feats, split_edges, n_info \
            = load_data(dataset, run=run, no_supra=config.no_gnn, prep_dir=config.prep_dir)

        n, n_layers, directed, mpx, _, p = n_info
        epochs = config.epochs
        input_dim = None
        if feats is not None:
            input_dim = feats.shape[1]
            feats = feats.to(device)

        model = build_model(config, n_layers, input_dim)
        dim = config.hidden_dim
        if config.heads_mode == 'concat':
            dim = dim * config.n_heads

        if device.type == 'cuda':
            if not config.no_gnn and g_supra is not None:
                g_supra = g_supra.to(device)
            for l_id in range(n_layers):
                g_train[l_id] = g_train[l_id].to(device)
                g_train_pos[l_id] = g_train_pos[l_id].to(device)
                g_test_pos[l_id] = g_test_pos[l_id].to(device)
                g_test_neg[l_id] = g_test_neg[l_id].to(device)
                g_val_pos[l_id] = g_val_pos[l_id].to(device)
                g_val_neg[l_id] = g_val_neg[l_id].to(device)
            model = model.to(device)

        if config.no_gnn:
            pars = model.parameters()
            predictor = None
        else:
            op = config.predictor
            predictor = [get_predictor(op, config, dim).to(device) for _ in range(n_layers)]
            pars = list(model.parameters())
            for i in range(len(predictor)):
                pars = pars + list(predictor[i].parameters())

        optimizer = optim.Adam(pars, lr=config.lr, weight_decay=config.weight_decay)
        stopper = EarlyStopping(patience=100, maximize=True,
                                       model_name=config.dataset + str(seed) + "_" + str(run),
                                       model_dir=config.ck_dir)

        t_total = time.time()
        for epoch in tqdm(range(1, epochs+1)):
            model.train()
            if predictor:
                for lid in range(n_layers):
                    predictor[lid].train()
            optimizer.zero_grad()

            g_train_negs = generate_negative_samples(g_train_pos, n_layers, n)

            if config.no_gnn or config.no_struct:
                pos_score, _, _ = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn)
                neg_score, _, _ = model(g_supra, g_train, p, g_train_negs, feats, predictor, omn)
                pos_score = torch.vstack(pos_score)
                neg_score = torch.vstack(neg_score)
                loss = compute_loss(pos_score, neg_score, device=device)
            else:
                pos_score, pos_struct, pos_gnn = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn)
                neg_score, neg_struct, neg_gnn = model(g_supra, g_train, p, g_train_negs, feats, predictor, omn)
                pos_score = torch.vstack(pos_score)
                neg_score = torch.vstack(neg_score)
                pos_struct = torch.vstack(pos_struct)
                neg_struct = torch.vstack(neg_struct)
                pos_gnn = torch.vstack(pos_gnn)
                neg_gnn = torch.vstack(neg_gnn)
                loss2 = compute_loss(pos_struct, neg_struct, device=device)
                loss3 = compute_loss(pos_gnn, neg_gnn, device=device)
                loss1 = compute_loss(pos_score, neg_score, device=device)
                loss = lambda1*loss1 + lambda2 * loss2 + lambda3*loss3

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not config.no_gnn:
                for lid in range(n_layers):
                    torch.nn.utils.clip_grad_norm_(predictor[lid].parameters(), 1.0)

            optimizer.step()

            if epoch > epoch_val:
                results = compute_score(pos_score.detach().cpu(), neg_score.detach().cpu(), scores=['ap', 'auc'])
                train_auc = results['auc']
                train_ap = results['ap']
                results = eval(g_supra, g_train, g_test_pos,
                               g_test_neg, g_val_pos, g_val_neg,
                               p, feats, omn, model, predictor)
                print(
                    'Epoch {:05d} | loss {:.4f} | valid auc {:.4f} | valid ap {:.4f} | train auc {:.4f} '
                    '| train ap {:.4f}'.
                    format(epoch, loss, results['auc'][0], results['ap'][0], train_auc, train_ap))

                stopper.step(results['auc'][0], model, epoch)  # Validation AUC
                if stopper.counter == 0 and predictor is not None:
                    for lid in range(n_layers):
                        torch.save(predictor[lid].state_dict(), os.path.join(stopper.model_dir,
                                                                        f"predictor{lid+1}_" + config.dataset + str(seed) + "_" + str(
                                                                            run) + ".bin"))
                for key, result in results.items():
                    loggers[key].add_result(run, result)

            else:
                print('Epoch {:05d} | loss: {:.4f}'.format(epoch, loss))

        tot_time = time.time() - t_total
        print("Optimization finished")
        print("Total training time: {:.4f}s".format(tot_time))
        for key in loggers.keys():
            v_best, t_best = loggers[key].print_statistics(run)
            print('Run {} - Final valid {} : {:.3f} '.format(run + 1, key, v_best) )
        print()

        model.load_state_dict(torch.load(stopper.save_dir))
        model = model.to(device)
        model.eval()
        if predictor is not None:
            for lid in range(len(predictor)):
                predictor[lid].load_state_dict(torch.load(os.path.join(stopper.model_dir, f"predictor{lid+1}_" + config.dataset + str(seed) + "_" + str(run) + ".bin")))
                predictor[lid] = predictor[lid].to(device)
                predictor[lid].eval()

        pos_scores, neg_scores = eval(g_supra, g_train, g_test_pos,
                                      g_test_neg, g_val_pos, g_val_neg,
                                      p, feats, omn, model, predictor, return_scores=True)

        all_pos_scores.append(pos_scores)
        all_neg_scores.append(neg_scores)

        if predictor is not None:
            for lid in range(len(predictor)):
                os.remove(
                        os.path.join(stopper.model_dir, f"predictor{lid+1}_" + config.dataset + str(seed) + "_" + str(
                            run) + ".bin"))

        stopper.remove_checkpoint()

        del model, predictor

        del g_supra, g_train, g_train_pos, g_test_pos, g_test_neg, g_val_pos, g_val_neg, feats, split_edges, n_info

    res = {}
    pos_scores = torch.vstack(all_pos_scores)
    neg_scores = torch.vstack(all_neg_scores)
    results = compute_score(pos_scores, neg_scores)
    res['auc'] = results['auc'] * 100
    res['ap'] = results['ap'] * 100
    print('Test results:')
    print('AUC : {:.3f} '.format(res['auc']))
    print('AP : {:.3f} '.format(res['ap']))

    write_results(config, res, name=f'results')  # Save final results


if __name__ == '__main__':
    args = set_params()
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.gpu < 0
    print_arguments(args)
    if cuda:
        torch.cuda.set_device(args.gpu)
        print('GPU device {}'.format(torch.cuda.get_device_name(args.gpu)))
        device = torch.device("cuda:" + str(args.gpu))
    else:
        print(f'Using CPU')
        device = torch.device("cpu")

    args.device = device
    train(args)


