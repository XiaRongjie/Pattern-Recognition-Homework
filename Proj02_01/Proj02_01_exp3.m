%%----------------Proj02-01：最小错误率贝叶斯分类器--------------%%
%%----------------Proj02-01-exp3----------------%%
%%协方差矩阵不变，但类均值向量分别变为m1 = (1, 3)T，m2 = (4, 0)T，重新进行实验；
clear; clc;
m1 = [1; 3]; m2 = [4; 0]; %均值
S1 = [1.5 0; 0 1]; S2 = [1 0.5; 0.5 2]; %协方差矩阵
n = 100; %随机样本数量为100
P1 = mvnrnd(m1, S1, n); %第一类样本
P2 = mvnrnd(m2, S2, n); %第二类样本
subplot(1, 2, 1);
s1 = scatter(P1(:, 1), P1(:, 2), '.');
hold on; s2 = scatter(P2(:, 1), P2(:, 2), 'v');
title('两类样本的二维散点图');
legend([s1 s2], '第一类样本点', '第二类样本点');

P = [P1; P2]; %两类样本
p1 = mvnpdf(P, m1', S1); %条件概率
p2 = mvnpdf(P, m2', S2);
p_w1 = 0.5; p_w2 = 0.5; %先验概率
g1 = p1 .* p_w1; %分类器函数
g2 = p2 .* p_w2;

%%——————最小错误率贝叶斯分类器——————%%
g = g1 - g2; 
D = zeros(size(P, 1), 1);
D(find(g > 0)) = 1;
D(find(g < 0)) = 2;
D(find(g == 0)) = inf;
%%———————————————————————%%

label = [ones(size(P1, 1), 1); 2 * ones(size(P2, 1), 1)]; %200个样本的标签
correct_id = find(D - label == 0);
wrong_id = find(abs(D - label) == 1);
unsure_id = find(D - label == inf);

correct = P(correct_id, :);
wrong = P(wrong_id, :);
unsure = P(unsure_id, :);

accuracy = size(correct, 1) / size(P, 1);

subplot(1, 2, 2);
c = scatter(correct(:, 1), correct(:, 2), 'o');
hold on; w = scatter(wrong(:, 1), wrong(:, 2), 'x');
acc = sprintf('正确分类百分比 = %f', accuracy);
st = ['样本点的分类结果:', string(acc)];
title(st);
legend([c w], '正确分类样本点', '错误分类样本点');
if ~isempty(unsure)
    u = scatter(unsure(:, 1), unsure(:, 2), '*');
    legend([c w u], '正确分类样本点', '错误分类样本点', '随机分类样本点');
end
%%
%%改变先验概率，利用贝叶斯分类器对相同样本进行重新分类
p_w11 = 0.4; p_w22 = 0.6; %先验概率
g11 = p1 .* p_w11; %分类器函数
g22 = p2 .* p_w22;

%%——————最小错误率贝叶斯分类器——————%%
gg = g11 - g22; 
DD = zeros(size(P, 1), 1);
DD(find(gg > 0)) = 1;
DD(find(gg < 0)) = 2;
DD(find(gg == 0)) = inf;
%%———————————————————————%%

label = [ones(size(P1, 1), 1); 2 * ones(size(P2, 1), 1)]; %200个样本的标签
correct_id1 = find(DD - label == 0);
wrong_id1 = find(abs(DD - label) == 1);
unsure_id1 = find(DD - label == inf);

correct1 = P(correct_id1, :);
wrong1 = P(wrong_id1, :);
unsure1 = P(unsure_id1, :);

accuracy1 = size(correct1, 1) / size(P, 1);

figure;
cc = scatter(correct1(:, 1), correct1(:, 2), 'o');
hold on; ww = scatter(wrong1(:, 1), wrong1(:, 2), 'x');
acc1 = sprintf('正确分类百分比 = %f', accuracy1);
st1 = ['样本点的分类结果:', string(acc1)];
title(st1);
legend([cc ww], '正确分类样本点', '错误分类样本点');
if ~isempty(unsure1)
    uu = scatter(unsure1(:, 1), unsure1(:, 2), '*');
    legend([cc ww uu], '正确分类样本点', '错误分类样本点', '随机分类样本点');
end
