import os
import numpy as np
import random
from collections import OrderedDict
import dataloaders
from torch.utils.data import DataLoader
import learners


class Trainer:
    """
    Trainer class for prompt-based continual learning

    Arguments:
    ----------
    args:
        Arguments from command line

    seed: int
        Random seed

    metric_keys: list
        List of metrics to be recorded

    save_keys: list
        List of metrics to be saved
    """
    def __init__(self, args, seed, metric_keys, save_keys):
        # Setttings
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.model_top_dir = args.log_dir  # model load directory

        self.top_k = 1  # top k accuracy
        
        # select dataset
        if args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32, 32, 3]
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # Shuffle class order
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')

        # split tasks
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p: p + inc])
            self.tasks_logits.append(class_order_logits[p: p + inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train')
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test')
        self.train_dataset = Dataset(
            args.dataroot, train=True, tasks=self.tasks,
            download_flag=True, transform=train_transform, 
            seed=self.seed, validation=args.validation
        )
        self.test_dataset  = Dataset(
            args.dataroot, train=False, tasks=self.tasks,
            download_flag=False, transform=test_transform, 
            seed=self.seed, validation=args.validation
        )

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {
            'num_classes': num_classes,
            'lr': args.lr,
            'debug_mode': args.debug_mode == 1,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'schedule': args.schedule,
            'schedule_type': args.schedule_type,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'gpuid': args.gpuid,
            'memory': args.memory,
            'temp': args.temp,
            'out_dim': num_classes,
            'overwrite': args.overwrite == 1,
            'DW': args.DW,
            'batch_size': args.batch_size,
            'upper_bound_flag': args.upper_bound_flag,
            'tasks': self.tasks_logits,
            'top_k': self.top_k,
            'prompt_param': [self.num_tasks, args.prompt_param]
        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)


    def task_eval(self, t_index, local=False, task='acc'):
        """
        Evaluate performance on task t_index

        Arguments:
        ----------
        t_index: int
            Task index

        local: bool
            Local evaluation on task t_index or global evaluation on all tasks

        task: str
            Task metric to evaluate
        """
        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)  # data and target is now task t_index
        test_loader  = DataLoader(
            self.test_dataset, batch_size=self.batch_size, 
            shuffle=False, drop_last=False, num_workers=self.workers, 
            pin_memory=True
        )
        if local:
            return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)
        

    def train(self, avg_metrics):
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: 
            temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): 
            os.makedirs(temp_dir, exist_ok=True)

        # for each task
        for i in range(self.max_task):
            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, 
                shuffle=True, drop_last=True, 
                num_workers=int(self.workers), pin_memory=True
            )

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(
                self.test_dataset, batch_size=self.batch_size, 
                shuffle=False, drop_last=False, 
                num_workers=self.workers, pin_memory=True
            )
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            prototype_save_dir = self.model_top_dir + '/prototypes/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): 
                os.makedirs(model_save_dir, exist_ok=True)
            if not os.path.exists(prototype_save_dir):
                os.makedirs(prototype_save_dir, exist_ok=True)

            avg_train_time = self.learner.learn_batch(
                train_loader, 
                self.train_dataset, 
                model_save_dir, 
                test_loader
            )

            # save model
            self.learner.save_model(model_save_dir)
            self.learner.save_prototype(prototype_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            prototype_save_dir = self.model_top_dir + '/prototypes/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'

            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()

            # load model
            self.learner.load_model(model_save_dir)
            self.learner.load_prototype(prototype_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics