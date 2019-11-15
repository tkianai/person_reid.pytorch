from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

from .engine import Engine
from reid.utils import AverageMeter, open_specified_layers, open_all_layers
from reid import metrics


class ImageEngine(Engine):

    def __init__(self, datamanager, model, optimizers, optimizer_weights=None, schedulers=None, use_gpu=True):
        super().__init__(datamanager, model, optimizers, schedulers, use_gpu)
        self.optimizer_weights = optimizer_weights

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch+1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers,
                                                          epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            for optimizer in self.optimizers:
                optimizer.zero_grad()
            total_loss, acc, loss_items = self.model(imgs, pids)
            total_loss.backward()

            for i, optimizer in enumerate(self.optimizers):
                # NOTE check consistence
                for param_group in optimizer.param_groups:
                    param_group['params'][0].grad.data *= self.optimizer_weights[i]

                optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(total_loss.item(), pids.size(0))
            accs.update(acc.item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * \
                    (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                          epoch+1, max_epoch, batch_idx+1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accs,
                          lr=self.optimizers[0].param_groups[0]['lr'],
                          eta=eta_str
                      )
                      )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Acc', accs.avg, n_iter)
                self.writer.add_scalar(
                    'Train/Lr', self.optimizers[0].param_groups[0]['lr'], n_iter)

            end = time.time()

        for scheduler in self.schedulers:
            scheduler.step()
