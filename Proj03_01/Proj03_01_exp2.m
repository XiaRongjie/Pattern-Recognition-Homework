%%----------------Proj03-01：主分量分析PCA--------------%%
%%--------------------Proj03-01-exp2-------------------%%
clc; clear;
N = 20;%样本数量
miu = [10; 15; 15];%均值 
sigma = [90, 2.5, 1.2; 2.5, 35, 0.2; 1.2, 0.2, 0.02];%协方差矩阵
X = mvnrnd(miu, sigma, N);%生成N个高斯分布的二维样本矢量
figure(1); plot3(X(:, 1), X(:, 2), X(:, 3), 'o'); title('样本集合X的三维散点图');
m = mean(X)';%样本集合X的均值向量
mm = repmat(m, 1, N);%%用矩阵的方法计算，repmat是复制和平铺矩阵
S = (X' - mm) * (X' - mm)';
S1 = (N - 1) * cov(X);
[V, D] = eig(S1);%D的对角线元素是特征值，V的列是相应的特征向量
[D_sort, index] = sort(diag(D),'descend');
V_sort = V(:,index);
% Y = V * (X' - mm);
Y1 = V_sort(:, 1)' * (X' - mm);
Y2 = V_sort(:, 2)' * (X' - mm);
Y = [Y1; Y2];
figure(2); plot(Y(1, :), Y(2, :), '+'); title('样本集合Y的二维散点图');
VV = inv(V);%求逆，这里使用没有排序之前的向量矩阵，才能得到垂直投影方向，用排序后的向量矩阵不能得到垂直投影方向
W = VV(:, 1:2);
Z = W * Y + mm;
figure(3); XX = plot3(X(:, 1), X(:, 2), X(:, 3), 'o'); 
hold on; ZZ = plot3(Z(1, :), Z(2, :), Z(3, :), '*'); 
legend([XX, ZZ], '集合X数据', '集合Z数据');
title('集合Z和集合X的三维数据散点图');
grid on;
for i = 1: N
    plot3([Z(1, i), X(i, 1)], [Z(2, i), X(i, 2)], [Z(3, i), X(i, 3)]);
end
grid on;
E = (X' - Z).^2;
Square_E = sum(sum(E));%计算所有的这些误差平方之和
MeanSquare_E = (1/N) * Square_E;%计算它们的均方误差
fprintf('误差平方之和 = %f\n', Square_E);
fprintf('均方误差 = %f\n', MeanSquare_E);