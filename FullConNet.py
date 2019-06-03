import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)
import numpy as np

how_much=101
class MLPNet:
    def __init__(self):
        #每个神经元只输出一个数据，对于每个神经元(w1x1+b1)+(w2x2+b2)......(w784x784+b784) == w1x1+w2x2+.......w784x784+(b1+b2+........+b784) ==  w1x1+w2x2+.......w784x784+B
        #输入矩阵[101 X 784],行向量为数据，行数为批次
        self.x = tf.placeholder(dtype=tf.float32,shape=[how_much,784])#图片张数
        # 输入矩阵[101 X 10],行向量为数据，行数为批次
        self.y = tf.placeholder(dtype=tf.float32,shape=[how_much,10])#y为ONE-HOT

        #在(u-2q,u+2q)上取值，标准差为0.1，值更集中在0附近，in_w的值更接近0
        #取输入层为100个神经元，每个神经元有784个W
        self.in_w = tf.Variable(tf.truncated_normal(shape=[784,100],stddev=0.1))
        # 取输入层为100个神经元，每个神经元有1个b，对784的每个数据乘以相应w相加后再加上b
        #取初始值为0
        self.in_b = tf.Variable(tf.zeros([100]))

        # 在(u-2q,u+2q)上取值，标准差为0.1，值更集中在0附近，in_b的值更接近0
        #输出层的上一层有100个神经元，也就是有100个输出，输出层有10个神经元，每个神经元对每个输入数据有1个w，一个神经元就有100个w
        self.out_w = tf.Variable(tf.truncated_normal(shape=[100,10],stddev=0.1))
        #输出层有10个神经元，所以有10个b
        self.out_b = tf.Variable(tf.zeros([10]))
    def forward(self):
        #输出矩阵【101 X 100】，每个行向量的每个元素为单个神经元对784个数据执行求和（wi*xi）再加b的结果，也就是单个神经元的输出
        #训练数据已进行过归一化处理（除以255），数据压缩到（0,1）之间
        self.fc1_f=tf.matmul(self.x,self.in_w)+self.in_b
        #小于0的取0，大于0的取原值
        #输出矩阵为【101 X 100】
        #如果采用sigmoid，w初始值趋近0，不存在梯度消失问题
        self.fc1 = tf.nn.relu(tf.matmul(self.x,self.in_w)+self.in_b)#转置？？？，已验证不需要转置,激活函数压缩,输入参数【101 x 784】X[784X100]=[101 X 100]
        #改1，效果不如原代码self.fc1 = tf.matmul(self.x, self.in_w) + self.in_b
        #改2，效果不如原代码self.fc1 = tf.nn.sigmoid(tf.matmul(self.x, self.in_w) + self.in_b)
        #输出矩阵【101 X 10】行向量为输出数据，列向量为批次
        self.output_f = tf.matmul(self.fc1,self.out_w)+self.out_b
        #对每个行向量执行softmax公式，每行相加为1
        #输出矩阵为【101 X 10】,行向量为每个神经元的输出结果，行数为批次
        #对每行执行softmax,值变到（0,1）之间，且每行的和为1
        self.output = tf.nn.softmax(tf.matmul(self.fc1,self.out_w)+self.out_b)#[101 X 100]（每个行向量的值都为正数） X [100 X 10]
    def backward(self):
        #把输出与one-hot形式比较
        self.loss = tf.reduce_mean((self.output-self.y)**2) #？？？
        self.opt = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
if __name__ == '__main__':
    net = MLPNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(100000):
            #每次取一百张，xs为数据，ys为标签
            xs,ys = mnist.train.next_batch(how_much)
            # test_output = sess.run(net.output,feed_dict={net.x:xs,net.y:ys})
            _loss, _, _output, _y, _output_f, fc1_f, in_w, netx = sess.run([net.loss,net.opt,net.output,net.y,net.output_f,net.fc1_f, net.in_w, net.x],feed_dict={net.x:xs,net.y:ys})

            if epoch % 1000 ==0:
                # print(_output_f[0])
                # print(_output[0])
                # print(_y[0])
                # print(_loss)
                # print(fc1_f[0])
                #print(_output[0].sum())

                test_xs,test_ys = mnist.test.next_batch(how_much)
                test_output = sess.run(net.output,feed_dict={net.x:test_xs})

                test_y = np.argmax(test_ys,axis=1)
                test_out = np.argmax(test_output,axis=1)
                print(np.mean(np.array(test_y == test_out,dtype=np.float32)))
                #print("标签",test_y,"结果",test_out)
                #print(sess.run(net.output))