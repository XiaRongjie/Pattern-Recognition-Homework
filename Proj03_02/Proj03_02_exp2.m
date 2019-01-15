%%----------------Proj03-02：Fisher线性判别分析FDA--------------%%
%%---------------Proj03-02-exp2 多重判别分析(MDA)---------------%%
clear; clc;
N = 10;
c = 3;
k = 1;%投影直线系数
%%第一类
w1 = [0.42 -0.087 0.58; -0.2 -3.3 -3.4; 1.3 -0.32 1.7; 0.39 0.71 0.23; -1.6 -5.3 -0.15; 
    -0.029 0.89 -4.7; -0.23 1.9 2.2; 0.27 -0.3 -0.87; -1.9 0.76 -2.1; 0.87 -1.0 -2.6];
%%第二类
w2 = [-0.4 0.58 0.089; -0.31 0.27 -0.04; 0.38 0.055 -0.035; -0.15 0.53 0.011; -0.35 0.47 0.034;
    0.17 0.69 0.1; -0.011 0.55 -0.18; -0.27 0.61 0.12; -0.065 0.49 0.0012; -0.12 0.054 -0.063];
%%第三类
w3 = [0.83 1.6 -0.014; 1.1 1.6 0.48; -0.44 -0.41 0.32; 0.047 -0.45 1.4; 0.28 0.35 3.1; 
    -0.39 -0.48 0.11; 0.34 -0.079 0.14; -0.3 -0.22 2.2; 1.1 1.2 -0.46; 0.18 -0.11 -0.49];
%%计算均值
m1 = mean(w1)';
m2 = mean(w2)';
m3 = mean(w3)';
m = (N * m1 + N * m2 + N * m3) / (3 * N);%%总体均值向量
%%计算类内散布矩阵
S1 = Intraclass_DM(w1, m1, N);
S2 = Intraclass_DM(w2, m2, N);
S3 = Intraclass_DM(w3, m3, N);

%%类别w1，w2和w3，计算最优方向矢量w
Sw = S1 + S2 + S3;
Sb = Interclass_DM(m1, m2, m3, m, N, c);%计算类间散布矩阵
S = inv(Sw) * Sb; 
[V, D] = eig(S);%%D的对角线元素是特征值，V的列是相应的特征向量
[D_sort, index] = sort(diag(D),'descend');
V_sort = V(:,index);
W1 = V_sort(:, 1); W2 = V_sort(:, 2);
W1 = W1 / norm(W1); W2 = W2 / norm(W2);%单位化
W = [W1 W2];
y1 = W' * w1';%%三个类别w1、w2和w3的所有数据点在二维子空间W上的投影
y2 = W' * w2';
y3 = W' * w3';
figure(1); WW1 = plot3(w1(:, 1), w1(:, 2), w1(:, 3), 'p');%%画w1,w2和w3的三维数据散点图
hold on; grid on; WW2 = plot3(w2(:, 1), w2(:, 2), w2(:, 3), 'o');
WW3 = plot3(w3(:, 1), w3(:, 2), w3(:, 3), '*'); 
legend([WW1, WW2, WW3], 'w1数据', 'w2数据', 'w3数据');
title('w1，w2和w3的三维数据散点图');
figure(2); Y1 = plot(y1(1, :), y1(2, :), '^');%%三个类别数据的二维散点图
hold on; grid on; Y2 = plot(y2(1, :), y2(2, :), '.');
Y3 = plot(y3(1, :), y3(2, :), '+');
legend([Y1, Y2, Y3], 'w1投影点', 'w2投影点', 'w3投影点');
title('w1，w2和w3在最优子空间中的投影点(MDA)');

%%计算投影后得到的三类二维数据的均值向量和协方差矩阵
miu1 = mean(y1')'; miu2 = mean(y2')'; miu3 = mean(y3')';
s1 = Cov(y1, miu1, N); s2 = Cov(y2, miu2, N); s3 = Cov(y3, miu3, N);
% s1 = cov(y1'); s2 = cov(y2'); s3 = cov(y3');%可以用这个函数计算协方差矩阵吗？
%%利用最小错误率贝叶斯分类器对投影后得到的三类二维数据集合进行分类
p_w1 = 1/3; p_w2 = 1/3; p_w3 = 1/3;%%贝叶斯分类器的先验概率
% [error_1,error_2,error_3,~,~,~]=Proj03_02_MADbayesclassify(y1',y2',y3',miu1', miu2', miu3', s1, s2, s3, p_w1, p_w2, p_w3);%坤
[f_max1, pre_b1] = Bayes_cla(y1, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
[f_max2, pre_b2] = Bayes_cla(y2, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
[f_max3, pre_b3] = Bayes_cla(y3, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
%%计算分类器的训练误差，即错分点的个数
label1 = ones(N, 1); label2 = 2 * ones(N, 1); label3 = 3 * ones(N, 1);%标签
error1 = length(find((pre_b1 - label1)~=0));%w1错误分类的个数
error2 = length(find((pre_b2 - label2)~=0));%w2错误分类的个数
error3 = length(find((pre_b3 - label3)~=0));%w3错误分类的个数
fprintf('在最优子空间中(MDA)，\n使用贝叶斯分类器对w1，w2和w3的所有数据点进行分类\n');
fprintf('------------w1样本点: 第%d类-------------\n', pre_b1);
fprintf('w1数据的训练误差：错分点为%d个\n\n', error1);
fprintf('------------w2样本点: 第%d类-------------\n', pre_b2);
fprintf('w2数据的训练误差：错分点为%d个\n\n', error2);
fprintf('------------w3样本点: 第%d类-------------\n', pre_b3);
fprintf('w3数据的训练误差：错分点为%d个\n\n', error3);

%% 对比实验:在非最优子空间中，使用贝叶斯分类器计算w2和w3数据的训练误差
v1 = [1.0 2.0 -1.5]'; v2 = [-1.0 0.5 -1.0]';
ww1 = v1 / norm(v1);%单位化
ww2 = v2 / norm(v2);
%%类别w2和w3的所有数据点在矢量方向w上的投影
W = [ww1 ww2];
yy1 = W' * w1';%%三个类别w1、w2和w3的所有数据点在二维子空间W上的投影
yy2 = W' * w2';
yy3 = W' * w3';
figure(3); YY1 = plot(yy1(1, :), yy1(2, :), '^');%%标记出投影后的点在直线上的位置 
hold on; grid on; YY2 = plot(yy2(1, :), yy2(2, :), '.');
YY3 = plot(yy3(1, :), yy3(2, :), '+');
legend([YY1, YY2, YY3], 'w1投影点', 'w2投影点', 'w3投影点');
title('w1，w2和w3在非最优子空间中的投影点');

%%计算投影后得到的三个二维数据的均值和方差
Miu1 = mean(yy1')'; Miu2 = mean(yy2')'; Miu3 = mean(yy3')';
ss1 = Cov(yy1, Miu1, N); ss2 = Cov(yy2, Miu2, N); ss3 = Cov(yy3, Miu3, N);
% ss1 = cov(yy1'); ss2 = cov(yy2'); ss3 = cov(yy3');%可以用这个函数计算协方差矩阵吗？
%%利用最小错误率贝叶斯分类器对训练类样本进行分类
p_w1= 1/3; p_w2 = 1/3; p_w3 = 1/3;%%贝叶斯分类器的先验概率
[F_max1, Pre_b1] = Bayes_cla(yy1, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
[F_max2, Pre_b2] = Bayes_cla(yy2, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
[F_max3, Pre_b3] = Bayes_cla(yy3, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
%%计算分类器的训练误差，即错分点的个数
Error1 = length(find((Pre_b1 - label1)~=0));%w1错误分类的个数
Error2 = length(find((Pre_b2 - label2)~=0));%w1错误分类的个数
Error3 = length(find((Pre_b3 - label3)~=0));%w1错误分类的个数
fprintf('\n在非最优子空间中，\n使用贝叶斯分类器对w1，w2和w3的所有数据点进行分类\n');
fprintf('------------w1样本点: 第%d类-------------\n', Pre_b1);
fprintf('w1数据的训练误差：错分点为%d个\n\n', Error1);
fprintf('------------w2样本点: 第%d类-------------\n', Pre_b2);
fprintf('w2数据的训练误差：错分点为%d个\n\n', Error2);
fprintf('------------w3样本点: 第%d类-------------\n', Pre_b3);
fprintf('w3数据的训练误差：错分点为%d个\n\n', Error3);

%% ---------------------子函数-------------------------- %%
function S = Intraclass_DM(x, m, N) %%计算类内散布矩阵；x为矩阵，m为向量，N为样本数目（标量）
S = zeros(size(m, 1));
for i = 1: N
    A = (x(i, :)' - m) * (x(i, :)' - m)';
    S = A + S; 
end
end

function Sb = Interclass_DM(m1, m2, m3, m, N, c) %%计算类间散布矩阵；x为矩阵，m为向量，c为样本类别数量
mm = [m1 m2 m3];
Sb = zeros(size(c, 1));
for i = 1: c
    A = N .* (mm(:, i) - m) * (mm(:, i) - m)';
    Sb = A + Sb; 
end
end

%%协方差矩阵计算函数
function S = Cov(x, m, N) %%x为矩阵，m为向量，N为样本数目（标量）
S = zeros(size(m, 1));
for i = 1: N
    A = (1 / N) .* ((x(:, i)' - m) * (x(:, i)' - m)');
    S = A + S; 
end
end

%%设计一个方向w上的一维贝叶斯分类器
function [f_max, pre_b] = Bayes_cla(x, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3)
N = size(x, 2);
p1 = zeros(N, 1); p2 = zeros(N, 1); p3 = zeros(N, 1);
g1 = zeros(N, 1); g2 = zeros(N, 1); g3 = zeros(N, 1);
f_max = zeros(N, 1); pre_b = zeros(N, 1);
for i = 1 : N
    p1(i) = mvnpdf(x(:, i)', m1', S1); %条件概率
    p2(i) = mvnpdf(x(:, i)', m2', S2);
    p3(i) = mvnpdf(x(:, i)', m3', S3);
    g1(i) = p1(i) .* p_w1; %分类器函数
    g2(i) = p2(i) .* p_w2;
    g3(i) = p3(i) .* p_w3;
    [f_max(i), pre_b(i)] = max([g1(i); g2(i); g3(i)]);
end
end
