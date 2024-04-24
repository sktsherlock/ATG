import torch
from torch.utils.data import DataLoader


def train(model, predictor, x, adj_t, edge_split, optimizer, batch_size):
    model.train()
    predictor.train()

    source_edge = edge_split['train']['source_node'].to(x.device)
    target_edge = edge_split['train']['target_node'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, x.size(0), src.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, edge_split, evaluator, batch_size, neg_len):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    def test_split(split, neg_length):
        source = edge_split[split]['source_node'].to(h.device)
        target = edge_split[split]['target_node'].to(h.device)
        target_neg = edge_split[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, neg_length).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, neg_length)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train', neg_len)
    valid_mrr = test_split('valid', neg_len)
    test_mrr = test_split('test', neg_len)


    return train_mrr, valid_mrr, test_mrr


def linkprediction(args, adj_t, edge_split, model, predictor, feat, evaluator, loggers, n_running, neg_len):
    # 定义优化器
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)
    # 进行训练
    for epoch in range(1, 1 + args.n_epochs):
        loss = train(model, predictor, feat, adj_t, edge_split, optimizer,
                     args.batch_size)

        if epoch % args.eval_steps == 0:
            results = test(model, predictor, feat, adj_t, edge_split, evaluator,
                           args.batch_size, neg_len=int(neg_len))
            for key, result in results.items():
                loggers[key].add_result(n_running, result)

            if epoch % args.log_every == 0:
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {n_running + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_hits:.2f}%, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%')
                print('---')

    return loggers
