import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from torch import nn
import pickle


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    '''

    :param model:
    :param optimizer:
    :param train_loader:
    :param model_func: takes `model` and `batch` as input, output `loss`,
    `tb_dict`, `disp_dict`, where `tb_dict` include expected log scalars.
    :param lr_scheduler:
    :param accumulated_iter:
    :param optim_cfg:
    :param rank:
    :param tbar:
    :param total_it_each_epoch:
    :param dataloader_iter:
    :param tb_log:
    :param leave_pbar:
    :return:
    '''
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    essential_module = model.module if isinstance(model,
                                                  nn.parallel.DistributedDataParallel) else model
    if hasattr(essential_module, 'split_parameters'):
        point_params, image_params = essential_module.split_parameters()

    else:
        point_params = model.parameters()



    pred_collection = {}
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()


        loss, tb_dict, disp_dict, pred_dicts = model_func(model, batch)
        pred_collection.update(pred_dicts)
        loss.backward()


        clip_grad_norm_(point_params, optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})


        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    if len(pred_collection) > 0:
        train_loader.dataset.fake_boxes = pred_collection
    return accumulated_iter

def train_one_epoch_da(model, optimizer, train_loader_org, train_loader_tar, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, loader_iter_org, loader_iter_tar, tb_log=None, leave_pbar=False):
    '''

    :param model:
    :param optimizer:
    :param train_loader:
    :param model_func: takes `model` and `batch` as input, output `loss`,
    `tb_dict`, `disp_dict`, where `tb_dict` include expected log scalars.
    :param lr_scheduler:
    :param accumulated_iter:
    :param optim_cfg:
    :param rank:
    :param tbar:
    :param total_it_each_epoch:
    :param dataloader_iter:
    :param tb_log:
    :param leave_pbar:
    :return:
    '''
    if total_it_each_epoch == 2 * min(len(train_loader_org), len(train_loader_tar)):
        loader_iter_org = iter(train_loader_org)
        loader_iter_tar = iter(train_loader_tar)


    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    essential_module = model.module if isinstance(model,
                                                  nn.parallel.DistributedDataParallel) else model
    if hasattr(essential_module, 'split_parameters'):
        point_params, image_params = essential_module.split_parameters()

    else:
        point_params = model.parameters()

    for cur_it in range(total_it_each_epoch):
        dataloader_iter, train_loader = (loader_iter_org, train_loader_org) if cur_it % 2 != 0 else (loader_iter_tar, train_loader_tar)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)
        batch['ignore_box_loss'] = train_loader.dataset.dataset_cfg.get('IGNORE_BOX_LOSS', False)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()


        loss, tb_dict, disp_dict, _ = model_func(model, batch)
        loss.backward()


        clip_grad_norm_(point_params, optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})


        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, smooth_model=None):
    accumulated_iter = start_iter
    if model.model_cfg.get('SELF_TRAINING', False) and start_epoch > 0:
        preds_path = ckpt_save_dir / 'last_preds.pkl'
        with open(preds_path, 'rb') as f:
            train_loader.dataset.fake_boxes = pickle.load(f)
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )
            if smooth_model is not None:
                smooth_model.update(model)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                if model.model_cfg.get('SELF_TRAINING', False):
                    preds_path = ckpt_save_dir / 'last_preds.pkl'
                    with open(preds_path, 'wb') as f:
                        pickle.dump(train_loader.dataset.fake_boxes, f)
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

def train_model_da(model, optimizer, train_loader_org, train_loader_tar, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler_org=None,train_sampler_tar=None,
                lr_warmup_scheduler=None, ckpt_save_interval=3, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = 2 * min(len(train_loader_tar), len(train_loader_org))
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader_org.dataset, 'merge_all_iters_to_one_epoch')
            train_loader_org.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)


            assert hasattr(train_loader_tar.dataset,
                           'merge_all_iters_to_one_epoch')
            train_loader_tar.dataset.merge_all_iters_to_one_epoch(merge=True,
                                                                  epochs=total_epochs)
            total_it_each_epoch = 2 * min(len(train_loader_tar),len(train_loader_org)) // max(total_epochs, 1)

        loader_iter_org = iter(train_loader_org)
        loader_iter_tar = iter(train_loader_tar)

        for cur_epoch in tbar:
            if train_sampler_org is not None:
                train_sampler_org.set_epoch(cur_epoch)
            if train_sampler_tar is not None:
                train_sampler_tar.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch_da(
                model, optimizer, train_loader_org, train_loader_tar, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                loader_iter_org=loader_iter_org,
                loader_iter_tar=loader_iter_tar,

            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
