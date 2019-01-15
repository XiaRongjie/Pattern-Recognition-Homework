%%----------------Proj05-01：感知器算法--------------%%
%%----------Proj05-01-exp1：批处理感知器算法-----------%%
clear; clc;
%%第一类
w1 = [0.1 1.1; 6.8 7.1; -3.5 -4.1; 2.0 2.7; 4.1 2.8; 
    3.1 5.0; -0.8 -1.3; 0.9 1.2; 5.0 6.4; 3.9 4.0];
%%第二类
w2 = [7.1 4.2; -1.4 -4.3; 4.5 0.0; 6.3 1.6; 4.2 1.9;
    1.4 -3.2; 2.4 -4.0; 2.5 -6.1; 8.4 3.7; 4.1 -2.2];
%%第三类
w3 = [-3.0 -2.9; 0.54 8.7; 2.9 2.1; -0.1 5.2; -4.0 2.2; 
    -1.3 3.7; -3.4 6.2; -4.1 3.4; -5.1 1.6; 1.9 5.1];
%%第四类
w4 = [-2.0 -8.4; -8.9 0.2; -4.2 -7.7; -8.5 -3.2; -6.7 -4.0; 
    -0.5 -9.2; -5.3 -6.7; -8.7 -6.4; -7.1 -9.7; -8.0 -6.3];
%%样本数据规范化
Y_w1 = [ones(size(w1, 1), 1), w1];
Y_w2 = [ones(size(w2, 1), 1), w2];
Y_w3 = [ones(size(w3, 1), 1), w3];
Y_w4 = [ones(size(w4, 1), 1), w4];

%% 在w1和w2的训练数据上实验
figure(1); title('训练数据w1和w2的分布');%显示训练数据w1和w2的分布
plot(w1(:, 1), w1(:, 2), 'o');
hold on; grid on; plot(w2(:, 1), w2(:, 2), '+');
xlabel('x_1'); ylabel('x_2');
legend('w1', 'w2');
%初始化
a = [0 0 0]'; eta = [0.1, 0.2, 0.4]; theta = 0.5; 
Y = [Y_w1; -Y_w2]';%两类，一类标为+1，另一类为-1

figure(2); hold on; title('在训练数据w1和w2，J_p(a)的值随迭代次数k的变化曲线');
for i = 1: size(eta, 2)
    [ak, Jp_a, k] = batch_perceptron(Y, a, eta(i), theta);
    if eta(i) == eta(1)
        plot(0: k-1, Jp_a, '-*');
    end
    if eta(i) == eta(2)
        plot(0: k-1, Jp_a, '-o');
    end
    if eta(i) == eta(3)
        plot(0: k-1, Jp_a, '-^'); 
    end
end
xlabel('k'); ylabel('J_p(a)');
legend('\eta = 0.2', '\eta = 0.4', '\eta = 0.6');
%% 分类
figure(3); title('训练数据w1和w2的分类');%显示训练数据w1和w2的分布
plot(w1(:, 1), w1(:, 2), 'o');
hold on; grid on; plot(w2(:, 1), w2(:, 2), '+');
xmin = min(min(w1(:,1)),min(w2(:,1)));
xmax = max(max(w1(:,1)),max(w2(:,1)));
xindex = xmin-1: (xmax-xmin)/100: xmax+1;
yindex = -ak(2)*xindex/ak(3) - ak(1)/ak(3);
plot(xindex, yindex);
xlabel('x_1'); ylabel('x_2');
legend('w1', 'w2', '决策面');
%% 
%% 在w2和w3的训练数据上实验
figure(4);  title('训练数据w2和w3的分布');%显示训练数据w2和w3的分布
plot(w2(:, 1), w2(:, 2), 'o');
hold on; grid on; plot(w3(:, 1), w3(:, 2), '+');
xlabel('x_1'); ylabel('x_2');
legend('w2', 'w3');
%初始化
a = [0 0 0]'; eta = [0.1, 0.2, 0.4]; theta = 0.5; 
Y = [Y_w2; -Y_w3]';%两类，一类标为+1，另一类为-1

figure(5); hold on; title('在训练数据w2和w3，J_p(a)的值随迭代次数k的变化曲线');
for i = 1: size(eta, 2)
    [ak, Jp_a, k] = batch_perceptron(Y, a, eta(i), theta);
    if eta(i) == eta(1)
        plot(0: k-1, Jp_a, '-*');
    end
    if eta(i) == eta(2)
        plot(0: k-1, Jp_a, '-o');
    end
    if eta(i) == eta(3)
        plot(0: k-1, Jp_a, '-^'); 
    end
end
xlabel('k'); ylabel('J_p(a)');
legend('\eta = 0.3', '\eta = 0.5', '\eta = 0.7');
%% 分类
figure(6); title('训练数据w1和w2的分类');%显示训练数据w1和w2的分布
plot(w2(:, 1), w2(:, 2), 'o');
hold on; grid on; plot(w3(:, 1), w3(:, 2), '+');
xmin = min(min(w2(:,1)),min(w3(:,1)));
xmax = max(max(w2(:,1)),max(w3(:,1)));
xindex = xmin-1: (xmax-xmin)/100: xmax+1;
yindex = -ak(2)*xindex/ak(3) - ak(1)/ak(3);
plot(xindex, yindex);
xlabel('x_1'); ylabel('x_2');
legend('w2', 'w3', '决策面');

%% 子函数
function [a, Jp_a, k] = batch_perceptron(Y, a, eta, theta)
%批处理感知器算法
%输入：Y为样本数据，a为初始化的权值向量，learning_rate为学习率，theta为阈值
%输出：a为权值向量，Jp_a为准则函数，k为收敛时的迭代步数
	k = 0;
    Jp_a = [];%准则函数
    while(1)
        k = k + 1;
        Yk = Y(:, a'*Y <= 0);%找出错分的点，形成集合Yk
        Jp_a = [Jp_a, sum(- a'*Yk, 2)];%计算准则函数
        d_Jp_a = sum(Yk, 2);
        a = a + eta * d_Jp_a;%更新权值
        if norm(eta * d_Jp_a) < theta
            break;
        end
    end
end
