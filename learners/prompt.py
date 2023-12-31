from __future__ import print_function

import torch
import numpy as np
import models
import copy

from .default import NormalNN
from utils.schedulers import CosineSchedule
from models.loss import ContrastivePrototypicalLoss
from utils.metric import AverageMeter, Timer


class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] in ['Adam', 'AdamW']:
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'cosine_torch':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule[-1], eta_min=1e-6)
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda', prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p', prompt_param=self.prompt_param)
        return model
    

class CPP(Prompt):
    def __init__(self, learner_config):
        super(CPP, self).__init__(learner_config)
        self.num_cls_per_task = learner_config['num_classes'] // 10
        self.batch_size = learner_config['batch_size']
        self.cl_neg = learner_config['cl_negative']
        self.criterion_fn = ContrastivePrototypicalLoss(
            temperature=learner_config['temp'],
            reduction="mean",
            cl_negative=self.cl_neg
        )

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'cpp', prompt_param=self.prompt_param)
        return model
    
    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters())
        print(f'Number of parameters: {sum(p.numel() for p in params_to_opt)}')
        
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] in ['Adam', 'AdamW']:
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'cosine_torch':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule[-1], eta_min=1e-6)
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)


    def update_model(self, inputs, targets):
        # Output embedding
        out, _ = self.model(inputs, train=True, pen=False)
        # Loss
        try:
            previous_prototype = self.model.sample_prototypes()
        except:
            previous_prototype = self.model.module.sample_prototypes()
        total_loss = self.criterion(out, targets.long(), previous_prototype)
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), out
    
    
    def criterion(self, embeddings, targets, previous_prototype):
        loss = self.criterion_fn(embeddings, targets, previous_prototype)
        return loss
    

    def predict(self, inputs):
        self.model.eval()
        out = self.model.predict(inputs)
        return out        # Output classes
    

    def forward(self, x, pen=False, train=False):
        return self.model.forward(x, pen=pen, train=train)
    

    def reset_model(self):
        # Reset MLP head
        self.model.prompt.reset_head()


    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            losses = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch
                if epoch > 0: 
                    self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, _ = self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    # record loss
                    y = y.detach()
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                    # print(f'Batches: {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time.sum:.2f}', end='\r')
                # print()
                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg}'.format(loss=losses))
                self.log(' * Total time {time:.2f}'.format(time=batch_time.sum))

                # reset
                losses = AverageMeter()

                # # Validation
                # if val_loader is not None and epoch % 3 == 0:
                #     self.task_count += 1
                #     self.update_prototypes(train_loader)
                #     self.validation(val_loader, task_metric='acc', verbal=True)
                #     self.task_count -= 1

                
        self.model.eval()
        # Update key and value prototypes
        self.update_prototypes(train_loader)

        self.first_task = False
        self.task_count += 1
        try:
            return batch_time.avg
        except:
            return None
        

    def update_prototypes(self, train_loader):
        self.model.eval()

        classes = np.zeros((0, ))
        query_feats = torch.zeros((0, 768))
        embeddings = torch.zeros((0, 768))

        for x, y, _  in train_loader:
            # send data to gpu
            if self.gpu:
                x = x.cuda()
            classes = np.concatenate((classes, y.cpu().numpy()))
            with torch.no_grad():
                try:
                    query = self.model.get_query_features(x).cpu()
                    out_features = self.model(x, pen=True, train=False).cpu()
                except AttributeError:
                    query = self.model.module.get_query_features(x).cpu()
                    out_features = self.model.module(x, pen=True, train=False).cpu()
            query_feats = torch.cat((query_feats, query), dim=0)
            embeddings = torch.cat((embeddings, out_features), dim=0)

        unique_classes = np.unique(classes)

        for cls in unique_classes:
            X_query = query_feats[classes == cls]
            X = embeddings[classes == cls]
            try:
                self.model.compute_key_prototypes(X_query, int(cls))
                self.model.update_value_prototypes(X, int(cls))
            except AttributeError:
                self.model.module.compute_key_prototypes(X_query, int(cls))
                self.model.module.update_value_prototypes(X, int(cls))



    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()

            with torch.no_grad():
                try:
                    output = model.predict(input)
                except:
                    output = model.module.predict(input)
                # print(self.task_count)
                assert output.max() <= self.task_count * self.num_cls_per_task
            acc_score = output.eq(target).float().mean()
            acc.update(acc_score, len(target))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg
    
    def save_prototype(self, filename):
        prototype = dict()
        try:
            prototype["key"] = copy.deepcopy(self.model.prompt.key_prototypes)
            prototype["value"] = copy.deepcopy(self.model.value_prototypes)
            prototype['std'] = copy.deepcopy(self.model.prototype_std)
        except:
            prototype["key"] = copy.deepcopy(self.model.module.prompt.key_prototypes)
            prototype["value"] = copy.deepcopy(self.model.module.value_prototypes)
            prototype['std'] = copy.deepcopy(self.model.module.prototype_std)
        self.log('=> Saving class prototype to:', filename)
        torch.save(prototype, filename + 'class.pth')
        self.log('=> Save Prototype Done')


    def load_prototype(self, filename):
        prototype = torch.load(filename + 'class.pth')
        self.model.prompt.key_prototypes = prototype["key"]
        self.model.value_prototypes = prototype["value"]
        self.model.prototype_std = prototype['std']
        self.log('=> Load Prototype Done')