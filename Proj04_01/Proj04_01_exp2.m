%%----------------Proj04-01：Parzen窗估计、k近邻估计--------------%%
%%------------------Proj04-01-exp1：k近邻估计---------------------%%
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
%% 采用欧氏距离度量，对具有n个训练样本点的一维数据，实现对任给测试点x的k近邻概率估计。
sample1 = w3(:, 1);
for k = [3, 5]%k参数
    n = 100;%测试点数量
    L = zeros(n, N);%欧式距离
    test = linspace(min(sample1) - 1, max(sample1) + 1, n);%测试点
    for i = 1: n
        for j = 1: N
            L(i, j) = sqrt((test(i) - sample1(j))^2);
        end
    end
    [a, b] = sort(L, 2);%对每个测试点到每个样本点之间的欧式距离进行排序，a为重新排好序，b为索引
    v1 = a(:, k);%%取前k列，即距离最近的k个，可视为体积V
    p = (k / N) ./ v1;
    if k == 3
        l1 = plot(test, p, '-');
    else
        l2 = plot(test, p, '--');
    end
    hold on;
end
legend([l1, l2], 'k = 3', 'k = 5');
xlabel('x'); ylabel('p(x)');
title('当k=3和5时，对一维概率密度函数的k近邻估计结果');

%% 采用欧氏距离度量，对具有n个训练样本点的二维数据，实现对任给测试点x的k近邻概率估计。
sample2 = w2(:, 1: 2);
for k = [3, 5]%k参数
    n = 100;%测试点数量
    L = zeros(n, N);%欧式距离
    test_x = linspace(min(sample2(:, 1)) - 1, max(sample2(:, 1)) + 1, n);%测试点
    test_y = linspace(min(sample2(:, 2)) - 1, max(sample2(:, 2)) + 1, n);
    [X, Y] = meshgrid(test_x, test_y);
    for i = 1: size(X(:), 1)
        for j = 1: N
            L(i, j) = sqrt((X(i) - sample2(j, 1))^2 + (Y(i) - sample2(j, 2))^2);%欧式距离
        end
    end
    [a, b] = sort(L, 2);%对每个测试点到每个样本点之间的欧式距离进行排序，a为重新排好序，b为索引
    v2 = pi * (a(:, k).^2);%%取前k列，即距离最近的k个，计算体积V
    p = (k / N) ./ v2;
    p = reshape(p, size(X));%将行向量转换为矩阵向量
    if k == 3
        figure; mesh(X, Y, p); title('当k=3时的k近邻估计的二维概率密度函数');
    else
        figure; mesh(X, Y, p); title('当k=5时的k近邻估计的二维概率密度函数');
    end
    xlabel('x1'); ylabel('x2'); zlabel('p(x)');
end

%% 采用欧氏距离度量，对已标记的具有三个类的三维训练数据，实现对任给测试点x的k近邻概率估计。
test1 = [-0.41, 0.82, 0.88]'; test2 = [0.14, 0.72, 4.1]'; test3 = [-0.81, 0.61, -0.38]';%测试点
w = cat(3, w1, w2, w3);
for k = [3, 5]%k参数
    class1 = KNN_classifier(w, test1, k);
    class2 = KNN_classifier(w, test2, k);
    class3 = KNN_classifier(w, test3, k);
    fprintf('--利用k近邻分类器(k=%d)对测试点进行分类--\n', k);
    fprintf('-------------测试点1: 第%d类-------------\n', class1);
    fprintf('-------------测试点2: 第%d类-------------\n', class2);
    fprintf('-------------测试点3: 第%d类-------------\n\n', class3);
end

%% 子函数
function class = KNN_classifier(w, test, k)
%%该函数用于k近邻算法分类
%%输入：w为训练样本集合，test为测试样本，k为k近邻参数
%%输出：class为k个近邻中出现最多的那个类别，即分类结果
    N = size(w, 1); c = size(w, 3);
    L = zeros(N, c);%欧式距离
    for i = 1: c
        for j = 1: N
%            L(j, i) = sqrt((test(1) - w(j, 1, i))^2 + (test(2) - w(j, 2, i))^2 + (test(3) - w(j, 3, i))^2);%欧式距离
%            L(j, i) = abs(test(1) - w(j, 1, i)) + abs(test(2) - w(j, 2, i)) + abs(test(3) - w(j, 3, i));%L1距离
            L(j, i) = max(max(abs(test(1) - w(j, 1, i)), abs(test(2) - w(j, 2, i))), abs(test(3) - w(j, 3, i)));%L无穷距离
        end 
    end
    t = sort(L(:));%对每个测试点到每个样本点之间的欧式距离进行排序，a为重新排好序，b为索引
    [~, n] = find(L <= t(k), k);%%寻找这k个点中属于各个类别的数量
    index = 1: max(n);
    h = histc(n, index);
    [~, max_index] = max(h);
    class = index(max_index);%%取最大数量的类别作为分类结果
end




