import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import datetime
import seaborn as sns
import gc
from xlutils.copy import copy
from matplotlib.font_manager import *
import os


RANGE = 5
force_gc = True

#垃圾回收
def collect_gc():
    if force_gc:
        gc.collect()


#生成具有一定条件限制的噪声
class NoiseDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        offset = np.random.random(N) * (float(self.range) / N)
        samples = np.linspace(-self.range, self.range, N) + offset
        return samples


#生成完全随机的噪声
def generate_noise(n):
    a = []
    for _ in range(n):
        a.append(np.random.random())
    return np.array(a)

#生成器网络
def generator(x, n_hidden=50, batch_size=1, timestep=50):
    #权值和偏置初始化
    w_init = tf.truncated_normal_initializer(stddev=2)
    b_init = tf.constant_initializer(0.)
    w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    #输入层
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.reshape(h0,[-1, timestep, n_hidden])#将输入至LSTM层的数据转换为三维张量
    #LSTM层
    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    init_state=cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states=tf.nn.dynamic_rnn(cell, h0, initial_state=init_state, dtype=tf.float32)
    output_rnn = tf.nn.relu(tf.reshape(output_rnn,[-1, n_hidden]))#将LSTM层计算的结果还原为二维张量
    #输出层
    w1 = tf.get_variable('w1', [output_rnn.get_shape()[1], 1], initializer=w_init)
    b1 = tf.get_variable('b1', [1], initializer=b_init)
    o = tf.matmul(output_rnn, w1) + b1
    return o


#鉴别器网络
def discriminator(x, n_hidden=50, batch_size=1, timestep=50):
    #权值和偏置初始化
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)
    w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    #输入层
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.reshape(h0,[-1, timestep, n_hidden])
    #LSTM层
    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    init_state=cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states=tf.nn.dynamic_rnn(cell, h0, initial_state=init_state, dtype=tf.float32)
    output_rnn = tf.nn.relu(tf.reshape(output_rnn,[-1, n_hidden]))
    #输出层
    w1 = tf.get_variable('w1', [output_rnn.get_shape()[1], 1], initializer=w_init)
    b1 = tf.get_variable('b1', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(output_rnn, w1) + b1)
    return o


#优化器
def optimizer(loss, var_list, num_decay_steps=400, initial_learning_rate=0.03):
    decay = 0.95
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


def main(my_sample, epoch, sample_size, hidden, batch_size, timestep):
    
    ''' 参数 '''
    B = sample_size         # 一次性学习的样本大小
    LR = 0.001              # 学习率
    TRAIN_ITERS = epoch     # 迭代次数
    n_hidden = hidden       # LSTM层神经元个数
    batch_size = batch_size # LSTM层的batch_size
    timestep = timestep     # LSTM层的timestep

    """ 构建图 """    

    #generator
    with tf.variable_scope('Gen'):
        z = tf.placeholder(tf.float32, shape=(None, 1))
        G_z = generator(z, n_hidden, batch_size, timestep)

    #discriminator
    with tf.variable_scope('Disc') as scope:
        x = tf.placeholder(tf.float32, shape=(None, 1))
        D_real = discriminator(x, n_hidden, batch_size, timestep)
        scope.reuse_variables()
        D_fake = discriminator(G_z, n_hidden, batch_size, timestep)

    # loss function
    loss_g = tf.reduce_mean(-tf.log(D_fake))
    loss_d = tf.reduce_mean(-tf.log(D_real) - tf.log(1 - D_fake))

    # trainable variables for each network
    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

    # optimizer for each network
    opt_d = optimizer(loss_d, d_params, 400, LR)
    opt_g = optimizer(loss_g, g_params, 400, LR)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    """ 交叉训练：训练D和G网络 """
    p_z = NoiseDistribution(RANGE)
    for step in range(TRAIN_ITERS):
        # 产生一个随机因子
        np.random.seed(np.random.randint(0, TRAIN_ITERS))
        
        # 训练D
        x_ = my_sample
        z_ = p_z.sample(B)
        loss_d_, _ = sess.run([loss_d, opt_d], {x: np.reshape(x_, (B, 1)), z: np.reshape(z_, (B, 1))})

        # 训练G
        z_ = p_z.sample(B)
        loss_g_, _ = sess.run([loss_g, opt_g], {z: np.reshape(z_, (B, 1))})

        if step % 1000 == 0:
            print('[%d/%d]: loss_d : %.3f, loss_g : %.3f' % (step, TRAIN_ITERS, loss_d_, loss_g_))


    """ 用训练好的G生成数据 """
    zs = p_z.sample(B)
    g = np.zeros((B, 1))
    for i in range(B // B):
        g[B * i:B * (i + 1)] = sess.run(G_z, {z: np.reshape(zs[B * i:B * (i + 1)], (B, 1))})
    g = g.tolist()
    for i in g:
        g_data.append(i)
    # 重置整个图，并回收垃圾
    tf.reset_default_graph()
    collect_gc()
    sess.close()


if __name__ == '__main__':
    
    sample_size = 288  # 一次性学习的样本大小
    hidden = 100       # LSTM层的神经元个数
    batch_size = 1     # LSTM层的batch_size
    timestep = 288     # LSTM层的timestep
    ''' sample_size = batch_size * hidden
        sample_size = batch_size * timestep '''
    all_start = datetime.datetime.now()
    # 迭代次数的list
    learn_set = [5000,5000,5000,5000,5000,5000,5000,5000,5000]
    flag = 0
    g_set = []
    for learn in learn_set:
        dirs = 'iteration_'+str(learn)+'batchsize_'+str(sample_size)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            
        g_data = []
        # 只学习300个样本点，方便展示效果
        df = pd.read_excel('parking.xls', usecols=[14])
        df = df[900:1188].values.tolist()
        for i in range(len(df)//sample_size):
            begin = datetime.datetime.now()
            data = df[i*sample_size:i*sample_size+sample_size]
            data = np.array(data)
            main(data, learn, sample_size, hidden, batch_size, timestep)
            end = datetime.datetime.now()
            print('[ '+str(i)+'  /  '+str(len(df)//sample_size)+' ]')
            print('用时: ', end-begin)
        g_set.append(g_data)  
        ''' 绘图展示结果 '''
        sns.set_style('darkgrid')
        #plt.subplot(111,axisbg='ghostwhite')
        plt.plot(df,'lightsteelblue',linewidth=3, label='real data')
        plt.plot(g_data,'lightcoral',linewidth=3, label='generated data')
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.xlabel('Time Point', fontsize=20)
        plt.ylabel('Unoccupied Parking Space Rate', fontsize=20)
        plt.title('Real Data and Generative Data', fontsize=21)
        plt.legend(fontsize=19)
        fig = plt.gcf()
        fig.set_size_inches(15,8)
        fig.savefig(dirs+'/'+str(flag)+'.png', dpi=100, bbox_inches='tight')
        plt.show()
        all_end = datetime.datetime.now()
        print('全部用时：', all_end - all_start)   
        flag += 1
        
    plt.subplot(331)
    plt.plot(df,'lightsteelblue',linewidth=3,label='real data')
    plt.plot(g_set[0],'lightcoral',linewidth=3,label='generative data')
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(fontsize=19)
    for i in range(1,len(g_set)):
        plt.subplot(331+i)
        plt.plot(df,'lightsteelblue',linewidth=3,label='real data')
        plt.plot(g_set[i],'lightcoral',linewidth=3,label='generative data')
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
    fig = plt.gcf()
    fig.set_size_inches(25,15)
    fig.savefig(dirs+'/9p.png', dpi=200, bbox_inches='tight')
    plt.show()