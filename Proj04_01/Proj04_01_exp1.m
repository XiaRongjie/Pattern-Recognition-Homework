%%----------------Proj04-01：Parzen窗估计、k近邻估计--------------%%
%%-----------------Proj04-01-exp1：Parzen窗估计-------------------%%
clear; clc;
N = 10;%每类样本数量
c = 3;%类别数目
%%第一类
w1 = [0.28 1.31 -6.2; 0.07 0.58 -0.78; 1.54 2.01 -1.63; -0.44 1.18 -4.32; -0.81 0.21 5.73; 
    1.52 3.16 2.77; 2.20 2.42 -0.19; 0.91 1.94 6.21; 0.65 1.93 4.38; -0.26 0.82 -0.96];
%%第二类
w2 = [0.011 1.03 -0.21; 1.27 1.28 0.08; 0.13 3.12 0.16; -0.21 1.23 -0.11; -2.18 1.39 -0.19;
    0.34 1.96 -0.16; -1.38 0.94 0.45; -0.12 0.82 0.17; -1.44 2.31 0.14; 0.26 1.94 0.08];
%%第三类
w3 = [1.36 2.17 0.14; 1.41 1.45 -0.38; 1.22 0.99 0.69; 2.46 2.19 1.31; 0.68 0.79 0.87; 
    2.51 3.22 1.35; 0.60 2.44 0.92; 0.64 0.13 0.97; 0.85 0.58 0.99; 0.66 0.51 0.88];
%%样本点
sample1 = [0.5, 1.0, 0.0]'; sample2 = [0.31, 1.51, -0.5]'; sample3 = [-0.3, 0.44, -0.1]';

%% 利用设计的基本Parzen窗估计分类器进行分类
p_w1 = 1/3; p_w2 = 1/3; p_w3 = 1/3;%%贝叶斯分类器的先验概率
for h = [1, 0.1]%parzen窗宽度
    [f_max1, pre_b1] = Bayes_classifier(sample1, w1, w2, w3, N, h, p_w1, p_w2, p_w3);%%贝叶斯分类器对样本点进行分类
    [f_max2, pre_b2] = Bayes_classifier(sample2, w1, w2, w3, N, h, p_w1, p_w2, p_w3);
    [f_max3, pre_b3] = Bayes_classifier(sample3, w1, w2, w3, N, h, p_w1, p_w2, p_w3);
    fprintf('--利用贝叶斯分类器对样本点进行分类(h=%.1f)--\n', h);
    fprintf('-------------样本点1: 第%d类----------------\n', pre_b1);
    fprintf('-------------样本点2: 第%d类----------------\n', pre_b2);
    fprintf('-------------样本点3: 第%d类----------------\n\n', pre_b3);
end
%% 利用设计的概率神经网络（PNN）分类器进行分类
for h = [1, 0.1]%parzen窗宽度
    sigma = h;%方差参数
    [weight, a] = PNN_train(w1, w2, w3);%%PNN训练
    class1 = PNN_classifier(sample1, weight, a, sigma);%%PNN分类器对样本点进行分类
    class2 = PNN_classifier(sample2, weight, a, sigma);
    class3 = PNN_classifier(sample3, weight, a, sigma);
    fprintf('--利用概率神经网络（PNN）分类器对样本点进行分类(h=%.1f)--\n', h);
    fprintf('---------------------样本点1: 第%d类---------------------\n', class1);
    fprintf('---------------------样本点2: 第%d类---------------------\n', class2);
    fprintf('---------------------样本点3: 第%d类---------------------\n\n', class3);
end

%% 子函数
function p_x_w = class_pdf(sample, w, N, h)
%%这个函数用于计算类条件概率密度
%%输入：sample为样本点，w为训练数据，N为训练数据数量，h为parzen窗宽度
%%输出：p_x_w为类条件概率密度
p_x_w = 0;
for i = 1: N
    parzen = exp((-(w(i, :)' - sample)' * (w(i, :)' - sample))/(2 * h^2));%窗函数，标量
    p_x_w = p_x_w + parzen;%%类条件概率密度
end
p_x_w = (1 / N) * p_x_w; 
end

function [f_max, pre_b] = Bayes_classifier(sample, w1, w2, w3, N, h, p_w1, p_w2, p_w3)
%%这个函数用于设计基本Parzen窗估计分类器：一个对三个类分类的贝叶斯分类器
%%输入：sample为样本点，w1、w2、w3为训练数据，N为训练数据数量，h为parzen窗宽度，p_w1、p_w2、p_w3为贝叶斯分类器先验概率
%%输出：f_max为最大判别函数值, pre_b为f_max对应的分类结果
p1 = class_pdf(sample, w1, N, h); %分别计算类条件概率
p2 = class_pdf(sample, w2, N, h);
p3 = class_pdf(sample, w3, N, h);
g1 = p1 * p_w1; %分类器函数
g2 = p2 * p_w2;
g3 = p3 * p_w3;
[f_max, pre_b] = max([g1; g2; g3]);
end

function [weight, a] = PNN_train(w1, w2, w3)
%%这个函数用于训练PNN网络的参数
%%输入：w1、w2、w3为训练数据
%%输出：w为训练得到的权重参数，a为索引
w1 = normr(w1); w2 = normr(w2); w3 = normr(w3);%归一化
weight1 = w1; weight2 = w2; weight3 = w3;%学习规则
weight = [weight1; weight2; weight3];
a = zeros(size(weight));
for i = 1: size(weight, 1)
    if i <= size(w1, 1)
        a(i, 1) = 1;
    end
    if i > size(w1, 1) && i < (size(w1, 1) + size(w2, 1))
        a(i, 2) = 1;
    end
    if i >= (size(w1, 1) + size(w2, 1))
        a(i, 3) = 1;
    end
end
end

function class = PNN_classifier(sample, weight, a, sigma)
%%这个函数用于PNN网络的分类
%%输入：sample为测试样本点，weight为训练权重参数，a为索引，sigma为方差
%%输出：class为分类结果
sample = normr(sample);%测试点归一化
net = zeros(size(weight, 1), 1);
g = zeros(size(weight));
for k = 1: size(net, 1)
    net(k, :) = weight(k, :) * sample;
    if a(k, 1) == 1
        g(k, 1) = g(k, 1) + exp((net(k) - 1) / (sigma^2));
    end
    if a(k, 2) == 1
        g(k, 2) = g(k, 1) + exp((net(k) - 1) / (sigma^2));
    end
    if a(k, 3) == 1
        g(k, 3) = g(k, 3) + exp((net(k) - 1) / (sigma^2));
    end
end
[~, class] = max(max(g));
end
