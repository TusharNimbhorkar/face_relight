import time,sys

import torch

from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
sys.path.append('models')
if __name__ == '__main__':
    opt = TrainOptions().parse()

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

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
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

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
                t_current = ((epoch_start_time.elapsed_time(iter_end_time) / 1000) / opt.batch_size)/epoch_iter
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data, {'Avg Time: %.3f':t_current})
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)


            if (total_steps+opt.batch_size) % opt.print_freq == 0:
                data_start_time.record()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        epoch_end_time.record()
        torch.cuda.synchronize(model.device)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,  epoch_start_time.elapsed_time(epoch_end_time)/1000))
        model.update_learning_rate()
