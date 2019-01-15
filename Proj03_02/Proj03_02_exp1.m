%%----------------Proj03-02：Fisher线性判别分析FDA--------------%%
%%--------------------Proj03-02-exp1-------------------%%
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

%%计算类内散布矩阵
S1 = Intraclass_DM(w1, m1, N);
S2 = Intraclass_DM(w2, m2, N);
S3 = Intraclass_DM(w3, m3, N);

%%类别w2和w3，计算最优方向矢量w
Sw = S2 + S3;%%计算总类内散布矩阵
w = inv(Sw) * (m2 - m3);
w = w / norm(w);%单位化
%%类别w2和w3的所有数据点在矢量方向w上的投影
y_2 = w' * w2';%%得到的是标量，即投影的长度
y_3 = w' * w3';
y2 = y_2' * w';%%投影长度乘上方向矢量，即为投影矢量
y3 = y_3' * w';
figure(1); W2 = plot3(w2(:, 1), w2(:, 2), w2(:, 3), 'o');%%画w2和w3的三维数据散点图
hold on; grid on; W3 = plot3(w3(:, 1), w3(:, 2), w3(:, 3), '*'); 
Line1 = plot3(k * [0 w(1)], k * [0 w(2)], k * [0 w(3)]);%%画表示方向矢量w的直线（投影方向直线）
Line2 = plot3(-k * [0 w(1)], -k * [0 w(2)], -k * [0 w(3)]);
Y2 = plot3(y2(:, 1), y2(:, 2), y2(:, 3), '.');%%标记出投影后的点在直线上的位置
Y3 = plot3(y3(:, 1), y3(:, 2), y3(:, 3), '+');
legend([W2, W3, Line1, Line2, Y2, Y3], 'w2数据', 'w3数据', '投影方向直线', '投影方向直线', 'w2投影点', 'w3投影点');
title('w2和w3的三维数据散点图及其在最优子空间中的投影点(FDA)');

%%计算投影后得到的两个一维数据的均值和方差
miu2 = mean(y2)'; miu3 = mean(y3)';
s2 = var(y2); s3 = var(y3);
%%利用最小错误率贝叶斯分类器对训练类样本进行分类
p_w2 = 0.5; p_w3 = 0.5;%%贝叶斯分类器的先验概率
[f_max2, pre_b2] = Bayes_cla(y2, miu2, miu3, s2, s3, p_w2, p_w3);
[f_max3, pre_b3] = Bayes_cla(y3, miu2, miu3, s2, s3, p_w2, p_w3);
%%计算分类器的训练误差，即错分点的个数
label2 = 2 * ones(N, 1); label3 = 3 * ones(N, 1);%标签
% error2 = sum(abs((pre_b2 + 1) - label2));%类别2正确分类的个数
error2 = length(find((pre_b2 + 1 - label2)~=0));%w2错误分类的个数
% error3 = sum(abs((pre_b3 + 1) - label3));%类别2正确分类的个数
error3 = length(find((pre_b3 + 1 - label3)~=0));%w3错误分类的个数
fprintf('在最优子空间中(FDA)，\n使用贝叶斯分类器对w2和w3的所有数据点进行分类\n');
fprintf('------------w2样本点: 第%d类-------------\n', pre_b2 + 1);
fprintf('w2数据的训练误差：错分点为%d个\n\n', error2);
fprintf('------------w3样本点: 第%d类-------------\n', pre_b3 + 1);
fprintf('w3数据的训练误差：错分点为%d个\n', error3);

% %%
% figure(3);
% % 一维高斯拟合
% [mu1, sigma1] = normfit(y_2);
% [mu2, sigma2] = normfit(y_3);
% x=-2:0.01:1.5;
% y2 = normpdf(x,mu1,sigma1);
% y3 = normpdf(x,mu2,sigma2);
% plot(x,y2,'r');
% hold on;
% plot(x,y3,'b');
% hold on;
% % 决策面
% if (mu2 > mu1)
% xt = mu1:0.01:mu2;
% else
%     if (mu2 < mu1)
%    xt = mu2:0.01:mu1;    
%     end
% end
% % 产生决策面
% y1 = normpdf(xt,mu1,sigma1);
% y2 = normpdf(xt,mu2,sigma2);
% judge = find(abs(y1-y2) < 0.05);
% judge = mu2 + 0.01 * judge(1);
% y=-1:0.1:3;
% x= judge + 0.*y; % 产生一个和y长度相同的x数组
% plot(x,y,'g.-');
% hold on;

%% 对比实验:在非最优子空间中，使用贝叶斯分类器计算w2和w3数据的训练误差
v = [1.0 2.0 -1.5]';
ww = v / norm(v);%单位化
%%类别w2和w3的所有数据点在矢量方向w上的投影
yy2 = ww' * w2';%%得到的是标量，即投影的长度
yy3 = ww' * w3';
yy2 = yy2' * ww';%%投影长度乘上方向矢量，即为投影矢量
yy3 = yy3' * ww';
figure(2); WW2 = plot3(w2(:, 1), w2(:, 2), w2(:, 3), 'o');%%画w2和w3的三维数据散点图
hold on; grid on; WW3 = plot3(w3(:, 1), w3(:, 2), w3(:, 3), '*'); 
line1 = plot3(2 * k * [0 ww(1)], 2 * k * [0 ww(2)], 2 * k * [0 ww(3)]);%%画表示方向矢量w的直线（投影方向直线）
line2 = plot3(-2 * k * [0 ww(1)], -2 * k * [0 ww(2)], -2 * k * [0 ww(3)]);
YY2 = plot3(yy2(:, 1), yy2(:, 2), yy2(:, 3), '.');%%标记出投影后的点在直线上的位置 
YY3 = plot3(yy3(:, 1), yy3(:, 2), yy3(:, 3), '+');
legend([WW2, WW3, line1, line2, YY2, YY3], 'w2数据', 'w3数据', '投影方向直线', '投影方向直线', 'w2投影点', 'w3投影点');
title('w2和w3的三维数据散点图及其在非最优子空间中的投影点');

%%计算投影后得到的两个一维数据的均值和方差
Miu2 = mean(yy2)'; Miu3 = mean(yy3)';
ss2 = var(yy2); ss3 = var(yy3);
%%利用最小错误率贝叶斯分类器对训练类样本进行分类
p_w2 = 0.5; p_w3 = 0.5;%%贝叶斯分类器的先验概率
[F_max2, Pre_b2] = Bayes_cla(yy2, Miu2, Miu3, ss2, ss3, p_w2, p_w3);
[F_max3, Pre_b3] = Bayes_cla(yy3, Miu2, Miu3, ss2, ss3, p_w2, p_w3);
%%计算分类器的训练误差，即错分点的个数
% Error2 = sum(abs((Pre_b2 + 1) - label2));%类别2正确分类的个数
Error2 = length(find((Pre_b2 + 1 - label2)~=0));%w2错误分类的个数
% Error3 = sum(abs((Pre_b3 + 1) - label3));%类别2正确分类的个数
Error3 = length(find((Pre_b3 + 1 - label3)~=0));%w3错误分类的个数
fprintf('\n在非最优子空间中，\n使用贝叶斯分类器对w2和w3的所有数据点进行分类\n');
fprintf('------------w2样本点: 第%d类-------------\n', Pre_b2 + 1);
fprintf('w2数据的训练误差：错分点为%d个\n\n', Error2);
fprintf('------------w3样本点: 第%d类-------------\n', Pre_b3 + 1);
fprintf('w3数据的训练误差：错分点为%d个\n', Error3);




%% ---------------------子函数-------------------------- %%
function S = Intraclass_DM(x, m, N) %%计算类内散布矩阵；x为矩阵，m为向量，N为样本数目（标量）
S = zeros(size(m, 1));
for i = 1: N
    A = (x(i, :)' - m) * (x(i, :)' - m)';
    S = A + S; 
end
end

%%设计一个方向w上的一维贝叶斯分类器
function [f_max, pre_b] = Bayes_cla(x, m1, m2, S1, S2, p_w1, p_w2)
N = size(x, 1);
p1 = zeros(N, 1); p2 = zeros(N, 1);
g1 = zeros(N, 1); g2 = zeros(N, 1);
f_max = zeros(N, 1); pre_b = zeros(N, 1);
for i = 1 : N
    p1(i) = mvnpdf(x(i, :), m1', S1); %条件概率
    p2(i) = mvnpdf(x(i, :), m2', S2);
    g1(i) = p1(i) .* p_w1; %分类器函数
    g2(i) = p2(i) .* p_w2;
    [f_max(i), pre_b(i)] = max([g1(i); g2(i)]);
end
end
