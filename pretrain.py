import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.utils import (
    build_args,
    set_random_seed,
    create_optimizer,
    create_loss,
    get_logger
)
from utils.data import load_data
from utils.dataset import BrainDataset
from utils.model import build_model_pretrain

    
def main():
    args = build_args()
    device = args.device if args.device >= 0 else "cpu"

    set_random_seed(args.seed)

    data = load_data(args)
    train_data = [data[idx] for idx in args.train_ids]
    test_data = [data[idx] for idx in args.test_ids]
    train_dataset = BrainDataset(train_data, args.spatial)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = build_model_pretrain(args)
    model = model.to(device)

    optimizer = create_optimizer(args.optim_type, model, args.lr, args.weight_decay)
    scheduler = lambda epoch :( 1 + np.cos((epoch - args.warmup) * np.pi / (args.max_epoch - args.warmup)) ) * 0.5 if epoch >= args.warmup else (epoch / args.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    criterion = create_loss(args.loss_fn)
    log_dir = os.path.join(f'../logs/{args.model}_pretrain', f'lr{args.lr}_weightdecay_{args.weight_decay}_cellmask_{args.cell_mask_rate}_genemask_{args.gene_mask_rate}_loss_{args.loss_fn}_optim_{args.optim_type}_gamma_{args.gamma}_ffn_{args.ffn_dim}_dropout_{args.dropout}_mp_{args.max_epoch}_trainids_{args.train_ids}_testids_{args.test_ids}_seed_{args.seed}')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(args.max_epoch):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device).squeeze(0)
            labels = batch[1].to(device).squeeze(0)
            if args.spatial:
                real_edge_mask = batch[2].to(device).squeeze(0)
                fake_edge_mask = batch[3].to(device).squeeze(0)

            if args.spatial:
                x_init, x_recon, encode_weights, embedding = model(inputs, real_edge_mask, fake_edge_mask)
            else:
                x_init, x_recon, encode_weights, embedding = model(inputs)
            loss = criterion(x_init, x_recon)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch + 1}: train_loss {train_loss / len(train_loader)}")
        writer.add_scalar('Train_Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        scheduler.step()
    
    # Test
    model.eval()
    if len(test_data) == 1:
        test_batch = test_data[0]
        inputs = test_batch['X']
        labels = test_batch['labels']
        if args.spatial:
            real_edge_mask = test_batch['real_edge_mask']
            fake_edge_mask = test_batch['fake_edge_mask']
        inputs = inputs.to(device)
        labels = labels.to(device)
        if args.spatial:
            real_edge_mask = real_edge_mask.to(device)
            fake_edge_mask = fake_edge_mask.to(device)

        if args.spatial:
            x_init, x_recon, encode_weights, embedding = model(inputs, real_edge_mask, fake_edge_mask)
        else:
            x_init, x_recon, encode_weights, embedding = model(inputs)
        loss = criterion(x_init, x_recon)
        logger.info(f"Test Loss: {loss.item()}")

        torch.save(model.state_dict(), os.path.join(log_dir, 'checkpoint.pt'))
        torch.save(encode_weights, os.path.join(log_dir, 'encode_weights.pt'))
        torch.save(embedding, os.path.join(log_dir, 'embedding.pt'))

    elif len(test_data) == 2:
        test_batch1 = test_data[0]
        inputs1 = test_batch1['X']
        if args.spatial:
            real_edge_mask1 = test_batch1['real_edge_mask']
            fake_edge_mask1 = test_batch1['fake_edge_mask']
        
        test_batch2 = test_data[1]
        inputs2 = test_batch2['X']
        if args.spatial:
            real_edge_mask2 = test_batch2['real_edge_mask']
            fake_edge_mask2 = test_batch2['fake_edge_mask']
        
        inputs = torch.cat((inputs1, inputs2), dim=0)
        inputs = inputs.to(device)
        if args.spatial:
            num_cell1 = inputs1.shape[0]
            num_cell2 = inputs2.shape[0]
            num_cell = inputs.shape[0] 

            real_edge_mask = torch.zeros((num_cell, num_cell))
            real_edge_mask[:num_cell1, :num_cell1] = real_edge_mask1
            real_edge_mask[num_cell1:, num_cell1:] = real_edge_mask2

            fake_edge_mask = torch.zeros((num_cell, num_cell))
            fake_edge_mask[:num_cell1, :num_cell1] = fake_edge_mask1
            fake_edge_mask[num_cell1:, num_cell1:] = fake_edge_mask2            

            real_edge_mask = real_edge_mask.bool().to(device)
            fake_edge_mask = fake_edge_mask.bool().to(device)

        if args.spatial:
            x_init, x_recon, encode_weights, embedding = model(inputs, real_edge_mask, fake_edge_mask)
        else:
            x_init, x_recon, encode_weights, embedding = model(inputs)

        torch.save(model.state_dict(), os.path.join(log_dir, 'checkpoint.pt'))
        torch.save(encode_weights, os.path.join(log_dir, 'encode_weights.pt'))
        torch.save(embedding, os.path.join(log_dir, 'embedding.pt'))

    else:
        torch.save(model.state_dict(), os.path.join(log_dir, 'checkpoint.pt'))


if __name__ == '__main__':
    main()