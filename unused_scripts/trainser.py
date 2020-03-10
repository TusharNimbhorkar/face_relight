import time,sys
from collections import OrderedDict

import torch

from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from commons.common_tools import Logger, BColors
from data.evaluation_dataset import EvaluationDataLoader
import os
sys.path.append('models')

log = Logger("General")

def validate(model, data_loader, epoch):
    '''
    Performs validation
    :param model: model to be evaluated
    :param data_loader: A data loader already set up with the validation set
    :param epoch: The current epoch
    :return:
    '''
    model.eval()
    n_batch_bk = model.opt.batch_size

    valid_start_time = torch.cuda.Event(enable_timing=True)
    valid_end_time = torch.cuda.Event(enable_timing=True)
    avg_losses = {}
    n_items = 0
    log_valid = Logger("Validation", tag_color=BColors.Red)
    log_valid.v("Validating...", replace=True)
    valid_start_time.record()

    for i, data in enumerate(data_loader):
        model.opt.batch_size = len(data['A'])
        model.set_input(data)


        with torch.no_grad():

            model.forward(epoch)
            model.backward_G(epoch, train_mode=False)
            losses = model.get_current_losses()
            for name in losses.keys():
                if name not in avg_losses.keys():
                    avg_losses[name] = 0
                avg_losses[name] += losses[name]

        n_items += len(data['A'])

        log_valid.v("Validating... %.2f %%" % (100*float(n_items)/len(data_loader)), replace=True)
            # model.optimize_parameters(0)

    print()

    for name in avg_losses.keys():
        avg_losses[name] = avg_losses[name]/n_items

    valid_end_time.record()
    torch.cuda.synchronize(model.device)

    visualizer.print_current_losses(epoch, 0, avg_losses, valid_start_time.elapsed_time(valid_end_time)/1000, 0, log=log_valid)
    model.opt.batch_size = n_batch_bk
    model.train_mode()

if __name__ == '__main__':
    opt = TrainOptions().parse()

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    data_root_dir = os.path.join(opt.dataroot, 'train')
    pair_list_path = opt.valid_list
    validate_n_batch = opt.valid_batch_size
    validation_data_loader = EvaluationDataLoader(data_root_dir, pair_list_path, validate_n_batch, opt.num_threads)


    print('#training images = %d' % dataset_size)


    def str_time(s):
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}'.format(int(hours), int(minutes))

    for epoch in range(1,15):
        epoch_start_time = torch.cuda.Event(enable_timing=True)
        epoch_end_time = torch.cuda.Event(enable_timing=True)

        iter_start_time = torch.cuda.Event(enable_timing=True)
        iter_end_time = torch.cuda.Event(enable_timing=True)

        data_start_time = torch.cuda.Event(enable_timing=True)
        data_end_time = torch.cuda.Event(enable_timing=True)

        epoch_start_time.record()
        data_start_time.record()


        epoch_iter = 0

        for i, data in enumerate(dataset):
            total_steps += opt.batch_size
            if total_steps % opt.print_freq == 0:
                iter_start_time.record()
                data_end_time.record()
            visualizer.reset()

            epoch_iter += opt.batch_size
            # print('set_input')
            model.set_input(data)
            # print('optimize')
            model.optimize_parameters(epoch)
            # print(model.named_parameters())
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                iter_end_time.record()
                torch.cuda.synchronize(model.device)
                t = (iter_start_time.elapsed_time(iter_end_time)/1000) / opt.batch_size
                t_data = (data_start_time.elapsed_time(data_end_time)/1000) / opt.batch_size
                t_current = (epoch_start_time.elapsed_time(iter_end_time) / 1000)/epoch_iter

                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data, ['Avg Time: %.3f' % t_current,
                                                                                       'Ep Time %s / %s' % (str_time(t_current*epoch_iter),str_time(t_current*(dataset_size)))])
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)


            if epoch % opt.valid_freq == 0:
                validate(model, validation_data_loader, epoch)

            if (total_steps+opt.batch_size) % opt.print_freq == 0:
                data_start_time.record()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        validate(model, validation_data_loader, epoch)

        epoch_end_time.record()
        torch.cuda.synchronize(model.device)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,  epoch_start_time.elapsed_time(epoch_end_time)/1000))
        model.update_learning_rate()
