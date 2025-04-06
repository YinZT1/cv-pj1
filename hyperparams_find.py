import model
import solver
import data_proc
import copy
data = data_proc.get_CIFAR10_data("./data/cifar-10-batches-py")

def find_best_hyper_params():
    '''
    测试了以下超参数，记录并输出最优的模型。
    '''
    lr_list = [1e-3,1e-2,0.005]
    lr_decay_list = [0.9,0.95,0.99]
    hidden_size_list = [64,100,256]
    reg_list = [1e-3,1e-4,1e-5]
    record = {}
    # lr_list = [1e-3]
    # lr_decay_list = [0.9]
    # hidden_size_list = [64]
    # reg_list = [1e-3]
    acc = 0.0
    best_solver = None
    for lr in lr_list:
        for lr_decay in lr_decay_list:
            for hidden_size in hidden_size_list:
                for reg in reg_list:
                    m = model.ThreeLayerModel(reg = reg,hidden_dim = hidden_size)
                    s = solver.Solver(data = data,model = m,update_rule='sgd',
                                    optim_config={
                                    'learning_rate': lr,
                                    },
                                    lr_decay=lr_decay,
                                    num_epochs=10, batch_size=100,
                                    print_every=100,
                                    verbose = False
                                    )
                    s.train()
                    record[(lr,lr_decay,hidden_size,reg)] = s.test()
                    if s.test()>acc:
                        acc = s.test()
                        best_solver = copy.deepcopy(s)
    return best_solver,record

