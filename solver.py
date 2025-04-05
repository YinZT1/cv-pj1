from __future__ import print_function, division
# from future import standard_library
# standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
import numpy as np
import optim
#使用dill代替pickle处理更复杂的保存checkpoint情况
import dill

class Solver(object):
    def __init__(self, data ,load_checkpoint_path = None, model = None,  **kwargs):
        #data必须传入，无论是否阅读了加载的模型。
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        #加载已经实现的模型
        self.load_checkpoint_path = load_checkpoint_path
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if self.load_checkpoint_path is not None:
            self._load_checkpoint(load_checkpoint_path)
            if model is not None and self.verbose:
                print("[Warning] 传入的model参数将被忽略，使用检查点中的模型")
            
        else:   
            self.model = model
            

            # Unpack keyword arguments
            self.update_rule = kwargs.pop('update_rule', 'sgd')
            self.optim_config = kwargs.pop('optim_config', {})
            self.lr_decay = kwargs.pop('lr_decay', 1.0)
            self.batch_size = kwargs.pop('batch_size', 100)
            self.num_epochs = kwargs.pop('num_epochs', 10)
            self.num_train_samples = kwargs.pop('num_train_samples', 1000)
            self.num_val_samples = kwargs.pop('num_val_samples', None)

        
        

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if isinstance(self.update_rule, str):
          if not hasattr(optim, self.update_rule):
              raise ValueError('Invalid update_rule "%s"' % self.update_rule)
          self.update_rule = getattr(optim, self.update_rule)
        else:
          assert callable(self.update_rule), 'The update rule is not callable'
        #进行判断，是否需要读取已有参数
        if self.load_checkpoint_path is None:
            self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        # optim_configs contains all params of optimization process:
        # W,b,learning rate,regular value
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            # pickle.dump(checkpoint, f)
            dill.dump(checkpoint,f)
        return checkpoint
    
    def _load_checkpoint(self,load_checkpoint_path):
        '''
        根据初始化kwargs的参数决定是否需要读取已经存在的模型
        参数包括：
        - model
        - update_rule
        - lr_decay
        - optim_config
        - batch_size
        - num_train_samples
        - num_val_samples
        - epoch 当前回合数
        - loss_history 记录了每个epoch的损失
        - train_acc_history 记录了每个epoch中训练集的准确度
        - val_acc_history 记录了每个epoch中验证集的准确度
        '''
        if load_checkpoint_path is None: return
        with open(load_checkpoint_path,'rb') as f:
            checkpoint = dill.load(f)
        #更新参数
        self.model = checkpoint['model']  # 强制要求 model 必须存在
        self.update_rule = checkpoint.get('update_rule', 'sgd')  # 提供默认值
        self.lr_decay = checkpoint.get('lr_decay', 1.0)
        self.optim_config = checkpoint.get('optim_config', {})
        self.batch_size = checkpoint.get('batch_size', 100)
        self.num_train_samples = checkpoint.get('num_train_samples', 0)
        self.num_val_samples = checkpoint.get('num_val_samples', 0)
        self.epoch = checkpoint['epoch']  # epoch 必须存在
        self.loss_history = checkpoint['loss_history']  # 必须存在
        self.train_acc_history = checkpoint.get('train_acc_history', [])
        self.val_acc_history = checkpoint.get('val_acc_history', [])

        if self.verbose:
            print(f'Loaded checkpoint from "{load_checkpoint_path}" (epoch {self.epoch})')

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def test(self,num_samples = None,batch_size = 100):
        '''
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        '''
        acc = self.check_accuracy(self.X_test,self.y_test,num_samples,batch_size)
        print(f"test dataset accuracy:{acc},test_data_size:{num_samples} ")
        return acc
    
