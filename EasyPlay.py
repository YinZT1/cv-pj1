import argparse
import model
import solver
import data_proc
import copy

data = data_proc.get_CIFAR10_data("./data/cifar-10-batches-py")

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l','--lr',type = float,help = '学习率',default=1e-3)
    parser.add_argument('-ld','--lrdecay',type = float,help = '学习率衰减',default=0.9)
    parser.add_argument('-hs','--hiddensize',type=int,help = '隐藏层大小',default=100)
    parser.add_argument('-r','--reg',type = float , help = '正则化系数',default=1e-3)
    parser.add_argument('-e','--NOfEpoch',type = int, help = '训练回合数' , default = 10)
    parser.add_argument('-b','--BatchSize',type= int , help= '批量',default=100)

    args = parser.parse_args()
    return args.lr,args.lrdecay,args.hiddensize,args.reg,args.NOfEpoch,args.BatchSize

if __name__ == '__main__':
    lr,lrdecay,hiddensize,reg,num_epoch,batch_size = get_params()
    print(lr,lrdecay,hiddensize,reg,num_epoch,batch_size)
    m = model.ThreeLayerModel(reg = reg,hidden_dim = hiddensize)
    s = solver.Solver(data = data,model = m,update_rule='sgd',
                                    optim_config={
                                    'learning_rate': lr,
                                    },
                                    lr_decay=lrdecay,
                                    num_epochs=num_epoch, batch_size=batch_size,
                                    print_every=100,
                                    verbose = True
                                    )
    s.train()
    print(f'''超参数如下：
        学习率：{lr}
        学习衰减：{lrdecay}
        隐藏层大小：{hiddensize}
        正则化系数：{reg}
        训练回合数：{num_epoch}
        批次大小：{batch_size}''')
    print(f'数据得到的结果是：{s.test()}')