

import os
import logging
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torchinfo import summary
from tqdm import tqdm


from loss_function_norm import calculate_loss, AverageMeter,\
    calculate_encode_loss, calculate_decode_loss
from models.MSSAE import EncoderBlock, MSSAE, DecoderBlock
from myDataset import CustomDataset, MyDataLoader
from utils import spectralSimilarity, experimentalResultsDisplay, create_logger

from setting import *


def dataAugment(hsi_origin, num_augment, batch_size, num_epochs, save_dir,
                device, type='b', num_cluster='all', num_data=2, window_size=3, load=False):
    in_len = hsi_origin.shape[-1]
    encoder = EncoderBlock(in_len=in_len, mapped_len=mapped_len, d_input=d_input, d_model=d_model, device=device)
    decoder = DecoderBlock(in_len=mapped_len, out_len=in_len, d_input=d_model, device=device)

    if load:
        encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'), map_location=device))
        decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pth'), map_location=device))

    AE_model = MSSAE(encoder, decoder)
    AE_model.train().to(device)
    optimizer = optim.Adam(AE_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)
    # 创建并配置独立的logger
    logger = create_logger(save_dir)
    hsi_augmented = []
    for epoch in range(num_epochs):
        dataLoader = DataLoader(hsi_origin, batch_size=batch_size, shuffle=True)
        print("Learning Rate:{}".format(optimizer.param_groups[0]['lr']))
        epoch_losses = AverageMeter()
        epoch_losses_sa = AverageMeter()
        epoch_losses_mse = AverageMeter()
        with tqdm(total=len(dataLoader)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, num_epochs))
            for x in dataLoader:
                x = x.to(device)
                encoded_out, decoded_out = AE_model(x)
                loss, loss_sa, loss_mse = calculate_decode_loss(x, decoded_out)
                epoch_losses.update(loss.item(), len(x))
                epoch_losses_sa.update(loss_sa.item(), len(x))
                epoch_losses_mse.update(loss_mse.item(), len(x))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))  # 记录损失值
                t.update(1)
        logger.info('epoch=%d, loss_sa=%.6f, loss_mse=%.6f',
                    epoch + 1, epoch_losses_sa.avg, epoch_losses_mse.avg)
        scheduler.step()
        # 生成数据
        if epoch_losses_sa.avg < 0.0022 and epoch_losses_mse.avg < 0.004:
            with torch.no_grad():
                for x in dataLoader:
                    x = x.to(device)
                    encoded_out, decoded_out = AE_model(x)
                    hsi_augmented += list(decoded_out.cpu().detach().numpy())
        if len(hsi_augmented) >= num_augment:
            # 保存生成的数据
            np.save(os.path.join(save_dir, '{}_{}_hsi_{}_{}_augmented.npy'.format(
                num_data, window_size, type, num_cluster)), np.asarray(hsi_augmented))
            torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'decoder.pth'))
            break

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(len(hsi_origin) // 1000 if type == 'b' else len(hsi_origin) // 10):
        plt.plot(hsi_origin[i])
    plt.title('Origin Spectral Spectrum')
    plt.savefig(os.path.join(save_dir, '{}_{}_hsi_{}_{}_origin.png'.format(num_data, window_size, type, num_cluster)))
    plt.close()
    plt.figure()
    for i in range(len(hsi_origin) // 1000 if type == 'b' else len(hsi_origin) // 10):
        plt.plot(hsi_augmented[i])
    plt.title('Augmented Spectral Spectrum')
    plt.savefig(os.path.join(save_dir, '{}_{}_hsi_{}_{}_augmented.png'.format(num_data, window_size, type, num_cluster)))
    plt.close()

    return torch.from_numpy(np.asarray(hsi_augmented))

def train(hsi, gt, HSI_DATA_NAME, hsi_b_tensor, hsi_t_tensor,
          ts_tensor, log_path, save_dir, device, load=False, alaf=0.2, shuffle=True, Binary_map=None):
    best_encoder_save_path = log_path + "best_encoder.pth"
    best_decoder_save_path = log_path + "best_decoder.pth"
    folder_display = os.path.dirname(best_encoder_save_path) + "/outputDisplay/"
    os.makedirs(folder_display, exist_ok=True)

    dataset = CustomDataset(hsi_b_tensor, hsi_t_tensor, ts_tensor, num_clusters=NUM_CLUSTERS)
    train_dataloader = MyDataLoader(dataset, batch_size=NUM_PER_CLUSTER, shuffle=shuffle)


    # test data
    hsi_tensor = torch.tensor(hsi, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(gt.reshape(-1), dtype=torch.int8, device=device)
    test_dataset = TensorDataset(hsi_tensor, label_tensor)
    test_iter = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    in_len = ts_tensor.shape[-1]
    encoder = EncoderBlock(in_len, mapped_len, d_input, d_model, layers=layers,
                        ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, device=device)
    decoder = DecoderBlock(mapped_len, in_len, d_model, d_input, use_unpooling=False, layers=layers,
                        ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, device=device)
    if load:
        encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'), map_location=device))
        decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pth'), map_location=device))

    AE_model = MSSAE(encoder, decoder)
    summary(AE_model, (64, in_len), device=device)
    # defining the optimizer
    optimizer = optim.Adam(AE_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // step_decay, gamma=gamma)


    # 创建并配置独立的logger
    logger = create_logger(log_path)
    best_auc_b = (0.0, 0.0, 0.0, 0.0, 0.0)
    best_auc = (0.0, 0.0, 0.0, 0.0, 0.0)
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        print(">>Epoch:{}".format(epoch))
        print("Learning Rate:{}".format(optimizer.param_groups[0]['lr']))
        epoch_ae_losses = AverageMeter()
        epoch_encoder_losses = AverageMeter()
        epoch_decoder_losses = AverageMeter()
        with tqdm(total=(len(train_dataloader))) as t:
            t.set_description('epoch: {}/{}'.format(epoch, epochs))
            encoder.train()
            decoder.train()
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                ts_tensor = ts_tensor.to(device)
                optimizer.zero_grad()
                similarity_coefficient = spectralSimilarity(x.squeeze(), ts_tensor, metric='SA')
                if MODEL_NAME == 'MSSAE':
                    encoded_x, decoded_x = AE_model(x)
                    _, encoded_ts = encoder(ts_tensor)
                    ae_loss, encoded_loss, decoded_loss = calculate_loss(
                                    x, encoded_x, encoded_ts, decoded_x, y,
                                    similarity_coefficient, encode_f=encode_f, alaf=alaf)
                    # ae_loss, encoded_loss, decoded_loss = calculate_loss(
                    #                 x, encoded_x, encoded_ts, decoded_x, y,
                    #                 similarity_coefficient, AE_model, encode_f=encode_f, alaf=alaf)
                    epoch_ae_losses.update(ae_loss.item(), len(x))
                    epoch_encoder_losses.update(encoded_loss.item(), len(x))
                    epoch_decoder_losses.update(decoded_loss.item(), len(x))
                    ae_loss.backward()
                    optimizer.step()
                    t.set_postfix(ae_loss='{:.6f}'.format(epoch_ae_losses.avg))  # 记录损失值
                    t.update(1)
                elif MODEL_NAME == 'MSSAE-ENCODER':
                    _, encoded_x = encoder(x)
                    _, encoded_ts = encoder(ts_tensor)
                    encoded_loss = calculate_encode_loss(
                        encoded_x, encoded_ts, y, similarity_coefficient)
                    epoch_encoder_losses.update(encoded_loss.item(), len(x))
                    encoded_loss.backward()
                    optimizer.step()
                    t.set_postfix(loss='{:.6f}'.format(epoch_encoder_losses.avg))  # 记录损失值
                    t.update(1)
        if MODEL_NAME == 'MSSAE':
            logger.info('epoch=%d, encoder_loss_avg=%.6f, decoder_loss_avg=%.6f',
                        epoch, epoch_encoder_losses.avg, epoch_decoder_losses.avg)
        scheduler.step()
        with torch.no_grad():  # 确保不进行梯度计算以节省内存
            outputs = []  # 创建一个列表来存储每一次的输出 Tensor
            encoder.eval()
            for x, y in test_iter:
                _, mapped_x = encoder(x)
                outputs.append(mapped_x)  # 将每一次的输出添加到 outputs 列表中
            mapped_hsi = torch.cat(outputs, dim=0)  # 在第 0 维上拼接所有的输出 Tensor
            _, mapped_ts = encoder(ts_tensor)
        # result = CEM(final_output.squeeze(), ts_tensor)
        # detection_map = result.reshape(*gt.shape)
        similarity = spectralSimilarity(mapped_hsi.squeeze(), mapped_ts, metric='SA')
        detection_map = similarity.reshape(*gt.shape)
        # auc = experimentalResultsDisplay(hsi, gt, np.array(detection_map.cpu()))
        # if auc > best_auc:
        #     # save the model after the last iteration
        #     experimentalResultsDisplay(hsi, gt, np.array(detection_map.cpu()),
        #                     folder_display=folder_display, dataset_name=HSI_DATA_NAME)
        #     torch.save(encoder.state_dict(), best_encoder_save_path)
        #     torch.save(decoder.state_dict(), best_decoder_save_path)
        #     best_auc = auc
        #     best_epoch = epoch
        # logger.info("Test AUC: {} (Best epoch: {}, Best Test AUC: {})".format(
        #     auc, best_epoch, best_auc))
        auc_b = experimentalResultsDisplay(hsi, Binary_map, np.array(detection_map.cpu()))
        auc = experimentalResultsDisplay(hsi, gt, np.array(detection_map.cpu()))
        if auc_b[0] > best_auc_b[0]:
            # save the model after the last iteration
            experimentalResultsDisplay(hsi, gt, np.array(detection_map.cpu()),
                            folder_display=folder_display, dataset_name=HSI_DATA_NAME)
            torch.save(encoder.state_dict(), best_encoder_save_path)
            torch.save(decoder.state_dict(), best_decoder_save_path)
            best_auc_b = auc_b
            best_auc = auc
            best_epoch = epoch
        logger.info("Test AUC_B: {} (Best epoch: {},Best Test AUC_B: {})".format(
            auc_b, best_epoch, best_auc_b))
        logger.info("Test AUC: {} (Best epoch: {}, Best Test AUC: {})".format(
            auc, best_epoch, best_auc))

