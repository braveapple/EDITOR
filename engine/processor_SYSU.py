import logging
import os
import time
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, Cross_R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("UniSReID.train")
    logger.info('start training')
    # Create SummaryWriter
    writer = SummaryWriter('/13559197192/wyh/UNIReID/runs/{}'.format(cfg.OUTPUT_DIR.split('/')[-1]))

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    evaluator_c1 = Cross_R1_mAP_eval(num_query, max_rank=20, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_c1.reset()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()

    best_index_c1 = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    best_index_c2 = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator_c1.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, imgpath) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view, img_path=imgpath,
                               writer=writer, epoch=epoch)
                loss = 0
                # 判断len(output)的奇偶性，如果是odd，那么执行下面的语句，如果是even，那么执行else语句
                if len(output) % 2 == 1:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    loss = loss + output[-1]
                else:
                    for i in range(0, len(output), 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
            writer.add_scalar('Loss', loss.item(), epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if output[0].shape[0] != target.shape[0]:
                target = torch.cat((target, target), dim=0)
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('!!!Mutil-Modal Testing!!!')
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            query_feat = model(img, cam_label=camids, view_label=target_view,
                                               cross_type=cfg.TEST.CROSS_TYPE, mode=0, img_path=_)
                            evaluator_c1.update_query((query_feat, vid, camid, _))
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(gallery_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            gallery_feat = model(img, cam_label=camids, view_label=target_view,
                                                 cross_type=cfg.TEST.CROSS_TYPE, mode=0, img_path=_)
                            evaluator_c1.update_gallery((gallery_feat, vid, camid, _))

                    if cfg.DATASETS.NAMES == 'RegDB' or cfg.DATASETS.NAMES == 'SYSU':
                        print('RegDB/SYSU Testing, No Multi-Modal!!!')
                    else:
                        pass

                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print(f'!!!Cross-Modal: All-Search  Testing!!!')
                    # 计算交叉模态性能
                    cmc_c1, mAP_c1, cmc_c1_verse, mAP_c1_verse = evaluator_c1.compute(cfg, epoch=epoch)
                    logger.info("[All] Validation Results ")
                    logger.info("mAP: {:.2%}".format(mAP_c1))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_c1[r - 1]))
                    writer.add_scalar('CM/All-Search/mAP', mAP_c1.item(), epoch)
                    writer.add_scalar('CM/All-Search/Rank-1', cmc_c1[0].item(), epoch)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print(f'!!!Cross-Modal: Indoor-Search Testing!!!')
                    logger.info("[All] Validation Results ")
                    logger.info("mAP: {:.2%}".format(mAP_c1_verse))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_c1_verse[r - 1]))
                    writer.add_scalar('CM/Indoor-Search/mAP', mAP_c1_verse.item(), epoch)
                    writer.add_scalar('CM/Indoor-Search/Rank-1', cmc_c1_verse[0].item(), epoch)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    # 保存最佳交叉模态模型
                    if mAP_c1 >= best_index_c1['mAP']:
                        best_index_c1['mAP'] = mAP_c1
                        best_index_c1['Rank-1'] = cmc_c1[0]
                        best_index_c1['Rank-5'] = cmc_c1[4]
                        best_index_c1['Rank-10'] = cmc_c1[9]
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best_All_Search.pth'))
                    logger.info("Best Cross-Modal All-Search mAP: {:.2%}".format(best_index_c1['mAP']))
                    logger.info("Best Cross-Modal All-Search Rank-1: {:.2%}".format(best_index_c1['Rank-1']))
                    logger.info("Best Cross-Modal All-Search Rank-5: {:.2%}".format(best_index_c1['Rank-5']))
                    logger.info("Best Cross-Modal All-Search Rank-10: {:.2%}".format(best_index_c1['Rank-10']))
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    # 保存最佳交叉模态模型
                    if mAP_c1_verse >= best_index_c2['mAP']:
                        best_index_c2['mAP'] = mAP_c1_verse
                        best_index_c2['Rank-1'] = cmc_c1_verse[0]
                        best_index_c2['Rank-5'] = cmc_c1_verse[4]
                        best_index_c2['Rank-10'] = cmc_c1_verse[9]
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best_Indoor_Search.pth'))
                    logger.info("Best Cross-Modal Indoor-Search mAP: {:.2%}".format(best_index_c2['mAP']))
                    logger.info("Best Cross-Modal Indoor-Search Rank-1: {:.2%}".format(best_index_c2['Rank-1']))
                    logger.info("Best Cross-Modal Indoor-Search Rank-5: {:.2%}".format(best_index_c2['Rank-5']))
                    logger.info("Best Cross-Modal Indoor-Search Rank-10: {:.2%}".format(best_index_c2['Rank-10']))
                    torch.cuda.empty_cache()
                    print('########################################')
            else:
                model.eval()
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('!!!Mutil-Modal Testing!!!')
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        query_feat = model(img, cam_label=camids, view_label=target_view,
                                           cross_type=cfg.TEST.CROSS_TYPE, mode=0, img_path=_)
                        evaluator_c1.update_query((query_feat, vid, camid, _))
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        gallery_feat = model(img, cam_label=camids, view_label=target_view,
                                             cross_type=cfg.TEST.CROSS_TYPE, mode=0, img_path=_)
                        evaluator_c1.update_gallery((gallery_feat, vid, camid, _))

                if cfg.DATASETS.NAMES == 'RegDB' or cfg.DATASETS.NAMES == 'SYSU':
                    print('RegDB/SYSU Testing, No Multi-Modal!!!')
                else:
                    pass

                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(f'!!!Cross-Modal: All-Search  Testing!!!')
                # 计算交叉模态性能
                cmc_c1, mAP_c1, cmc_c1_verse, mAP_c1_verse = evaluator_c1.compute(cfg, epoch=epoch)
                logger.info("[All] Validation Results ")
                logger.info("mAP: {:.2%}".format(mAP_c1))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_c1[r - 1]))
                writer.add_scalar('CM/All-Search/mAP', mAP_c1.item(), epoch)
                writer.add_scalar('CM/All-Search/Rank-1', cmc_c1[0].item(), epoch)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(f'!!!Cross-Modal: Indoor-Search Testing!!!')
                logger.info("[All] Validation Results ")
                logger.info("mAP: {:.2%}".format(mAP_c1_verse))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_c1_verse[r - 1]))
                writer.add_scalar('CM/Indoor-Search/mAP', mAP_c1_verse.item(), epoch)
                writer.add_scalar('CM/Indoor-Search/Rank-1', cmc_c1_verse[0].item(), epoch)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                # 保存最佳交叉模态模型
                if mAP_c1 >= best_index_c1['mAP']:
                    best_index_c1['mAP'] = mAP_c1
                    best_index_c1['Rank-1'] = cmc_c1[0]
                    best_index_c1['Rank-5'] = cmc_c1[4]
                    best_index_c1['Rank-10'] = cmc_c1[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best_All_Search.pth'))
                logger.info("Best Cross-Modal All-Search mAP: {:.2%}".format(best_index_c1['mAP']))
                logger.info("Best Cross-Modal All-Search Rank-1: {:.2%}".format(best_index_c1['Rank-1']))
                logger.info("Best Cross-Modal All-Search Rank-5: {:.2%}".format(best_index_c1['Rank-5']))
                logger.info("Best Cross-Modal All-Search Rank-10: {:.2%}".format(best_index_c1['Rank-10']))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                # 保存最佳交叉模态模型
                if mAP_c1_verse >= best_index_c2['mAP']:
                    best_index_c2['mAP'] = mAP_c1_verse
                    best_index_c2['Rank-1'] = cmc_c1_verse[0]
                    best_index_c2['Rank-5'] = cmc_c1_verse[4]
                    best_index_c2['Rank-10'] = cmc_c1_verse[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best_Indoor_Search.pth'))
                logger.info("Best Cross-Modal Indoor-Search mAP: {:.2%}".format(best_index_c2['mAP']))
                logger.info("Best Cross-Modal Indoor-Search Rank-1: {:.2%}".format(best_index_c2['Rank-1']))
                logger.info("Best Cross-Modal Indoor-Search Rank-5: {:.2%}".format(best_index_c2['Rank-5']))
                logger.info("Best Cross-Modal Indoor-Search Rank-10: {:.2%}".format(best_index_c2['Rank-10']))
                torch.cuda.empty_cache()
                print('########################################')

    writer.close()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("UniSReID.test")
    logger.info("Enter inferencing")

    if cfg.TEST.CROSS:
        evaluator = Cross_R1_mAP_eval(num_query, max_rank=20, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)
        model.eval()
        img_path_list = []

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                img = {'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device)}
                camids = camids.to(device)
                target_view = target_view.to(device)
                query_feat, gallery_feat = model(img, cam_label=camids, view_label=target_view,
                                                 cross_type=cfg.TEST.CROSS_TYPE, mode=0)
                evaluator.update_query((query_feat, pid, camid, imgpath))
                evaluator.update_gallery((gallery_feat, pid, camid, imgpath))
                img_path_list.extend(imgpath)

        cmc, mAP, _, _, _, _, _ = evaluator.compute(cfg)
        logger.info("Validation Results ")
        logger.info("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        return cmc[0], cmc[4]
    else:
        if cfg.DATASETS.NAMES == "MSVR310":
            evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator.reset()
        else:
            evaluator = R1_mAP_eval(num_query, max_rank=20, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator.reset()
        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)

        model.eval()
        img_path_list = []
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                img = {'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device)}
                camids = camids.to(device)
                scenceids = target_view
                target_view = target_view.to(device)
                feat = model(img, cam_label=camids, view_label=target_view)
                if cfg.DATASETS.NAMES == "MSVR310":
                    evaluator.update((feat, pid, camid, scenceids, imgpath))
                else:
                    evaluator.update((feat, pid, camid))
                img_path_list.extend(imgpath)

        cmc, mAP, _, _, _, _, _ = evaluator.compute(cfg)
        logger.info("Validation Results ")
        logger.info("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        return cmc[0], cmc[4]
