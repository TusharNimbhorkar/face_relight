import time,sys
from commons.torch_tools import Chronometer
import numpy as np

import torch
import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter

sys.path.append('models')
if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Reproducability
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir,opt.name,'logdir'))

    print(str(model.device))

    epoch_chron = Chronometer(model.device)
    iter_chron = Chronometer(model.device)
    data_chron = Chronometer(model.device)
    current_chron = Chronometer(model.device)
    elapsed_data = 0
    elapsed_iter = 0
    elapsed_epoch = 0
    elapsed_current = 0

    def str_time(s):
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}'.format(int(hours), int(minutes))
    if opt.ft:
        end_epoch_count = 5
    else:
        end_epoch_count = opt.end_epoch

    if opt.continue_train:
        start_epoch_count = int(opt.epoch) + 1
    else:
        start_epoch_count=1
    print(start_epoch_count)
    for epoch in range(start_epoch_count, end_epoch_count):
        epoch_chron.tick()
        data_chron.tick()
        current_chron.tick()


        epoch_iter = 0
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

        for i, data in enumerate(dataset):
            total_steps += opt.batch_size
            if total_steps % opt.print_freq == 0:
                iter_chron.tick()
                elapsed_data = data_chron.tock()
            visualizer.reset()

            epoch_iter += opt.batch_size
            model.set_input(data)
            if opt.ft:
                model.optimize_parameters()
            else:
                model.optimize_parameters(epoch)
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = iter_chron.tock() / opt.batch_size
                t_data = elapsed_data / opt.batch_size
                t_current = current_chron.tock()/epoch_iter

                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data, ['Avg Time: %.3f' % t_current,
                                                                                       'Ep Time %s / %s' % (str_time(t_current*epoch_iter),str_time(t_current*(dataset_size)))])

                for key in losses.keys():

                    writer.add_scalar(key,losses[key], total_steps)
                # writer.add_scalar(f'loss/check_info', {'G_GAN':losses['G_GAN'],'G_L1':losses['G_L1']}, total_steps)
                #epoch * len(trainloader) + i)


                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)


            if (total_steps+opt.batch_size) % opt.print_freq == 0:
                data_chron.tick()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        elapsed_epoch = epoch_chron.tock()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,  elapsed_epoch))
        model.update_learning_rate()
