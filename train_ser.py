import time,sys
from commons.torch_tools import Chronometer

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

    for epoch in range(1,15):
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
                t = iter_chron.tock() / opt.batch_size
                t_data = elapsed_data / opt.batch_size
                t_current = current_chron.tock()/epoch_iter

                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data, ['Avg Time: %.3f' % t_current,
                                                                                       'Ep Time %s / %s' % (str_time(t_current*epoch_iter),str_time(t_current*(dataset_size)))])
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
