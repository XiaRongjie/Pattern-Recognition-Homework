%%-------------------Proj06-01：BP神经网络--------------------%%
clear; clc;
%%第一类
w1 = [1.58 2.32 -5.8; 0.67 1.58 -4.78; 1.04 1.01 -3.63; -1.49 2.18 -0.39; -0.41 1.21 -4.73; 
    1.39 3.16 2.87; 1.20 1.40 -1.89; -0.92 1.44 -3.22; 0.45 1.33 -4.38; -0.76 0.84 -1.96];
%%第二类
w2 = [0.21 0.03 -2.21; 0.37 0.28 -1.8; 0.18 1.22 0.16; -0.24 0.93 -1.01; -1.18 0.39 -0.39;
    0.74 0.96 -1.16; -0.38 1.94 -0.48; 0.02 0.72 -0.17; 0.44 1.31 -0.14; 0.46 1.49 0.68];
%%第三类
w3 = [-1.54 1.17 0.64; 5.41 3.45 -1.33; 1.55 0.99 2.69; 1.86 3.19 1.51; 1.68 1.79 -0.87; 
    3.51 -0.22 -1.39; 1.40 -0.44 0.92; 0.44 0.83 1.97; 0.25 0.68 -0.99; -0.66 -0.45 0.08];

%% 构造一个3-3-1型的三层BP神经网络
X = [w1; w2];%BP网络输入层
t1 = 1; t2 = 0;%教师信号
T = [repmat(t1, 1, size(w1, 1)), repmat(t2, 1, size(w2, 1))];%输出层对应的样本点数个教师信号
  
eta = 0.1;%学习率
theta = 0.001; 
figure;  
for selection_rand = [1, 0]%%初始化各权值和偏置
    if selection_rand%随机初始化
        Wji = -1 + 2 * rand(3, 3);%输入层到中间层的权值矩阵
        Wjb = -1 + 2 * rand(3, 1);%中间层神经元的偏置向量
        Wkj = -1 + 2 * rand(1, 3);%中间层到输出层的权值矩阵
        Wkb = -1 + 2 * rand(1, 1);%输出层神经元的偏置向量
    else%初始化为固定值
        Wji = 0.5 * ones(3, 3);%输入层到中间层的权值矩阵
        Wjb = 0.5 * ones(3, 1);%中间层神经元的偏置向量
        Wkj = 0.5 * ones(1, 3);%中间层到输出层的权值矩阵
        Wkb = 0.5 * ones(1, 1);%输出层神经元的偏置向量
    end
    m = 0;
    J = [];
    while(1)
        m = m + 1;
        [Wji, Wjb, Wkj, Wkb, d_wkj, d_wji, Z1] = BP_train(eta, Wji, Wjb, Wkj, Wkb, X, T);
        J = [J, 1/2 * (norm(T - Z1))^2];
        if norm(d_wji) < theta && norm(d_wkj) < theta || m > 10000
            break;
        end
    end
    if selection_rand
        plot(J, '-'); fprintf('随机初始化权值的训练回合为：%d\n', m - 1);
    else
        plot(J, '--'); fprintf('固定初始化权值的训练回合为：%d\n', m - 1);
    end
    hold on; 
end
title('3-3-1型BP网络的损失函数'); legend('-1 <= W <= +1', 'W = 0.5');
xlabel('训练回合数'); ylabel('训练误差');

%% 构造一个3-4-3型的三层BP神经网络
X = [w1; w2; w3];%BP网络输入层
t1 = [1 0 0]; t2 = [0 1 0]; t3 = [0 0 1];%教师信号
T = [repmat(t1', 1, size(w1, 1)), repmat(t2', 1, size(w2, 1)), repmat(t3', 1, size(w3, 1))];%输出层对应的样本点数个教师信号
  
eta = 0.1;%学习率
theta = 0.0001; 
figure;  
for selection_rand = [1, 0]%%初始化各权值和偏置
    if selection_rand%随机初始化
        Wji = -1 + 2 * rand(4, 3);%输入层到中间层的权值矩阵
        Wjb = -1 + 2 * rand(4, 1);%中间层神经元的偏置向量
        Wkj = -1 + 2 * rand(3, 4);%中间层到输出层的权值矩阵
        Wkb = -1 + 2 * rand(3, 1);%输出层神经元的偏置向量
    else%初始化为固定值
        Wji = 0.5 * ones(4, 3);%输入层到中间层的权值矩阵
        Wjb = 0.5 * ones(4, 1);%中间层神经元的偏置向量
        Wkj = 0.5 * ones(3, 4);%中间层到输出层的权值矩阵
        Wkb = 0.5 * ones(3, 1);%输出层神经元的偏置向量
    end
    m = 0; 
    J = [];
    while(1)
        m = m + 1;
        [Wji, Wjb, Wkj, Wkb, d_wkj, d_wji, Z1] = BP_train(eta, Wji, Wjb, Wkj, Wkb, X, T);
        J = [J, 1/2 * (norm(T - Z1))^2];
        if norm(d_wji) < theta && norm(d_wkj) < theta || m > 1000
            break;
        end
    end
    if selection_rand
        plot(J, '-'); fprintf('随机初始化权值的训练回合为：%d\n', m - 1);
    else
        plot(J, '--'); fprintf('固定初始化权值的训练回合为：%d\n', m - 1);
    end
    hold on; 
end
title('3-4-3型BP网络的损失函数'); legend('-1 <= W <= +1', 'W = 0.5');
xlabel('训练回合数'); ylabel('训练误差');
