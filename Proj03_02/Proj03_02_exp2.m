%%----------------Proj03-02��Fisher�����б����FDA--------------%%
%%---------------Proj03-02-exp2 �����б����(MDA)---------------%%
clear; clc;
N = 10;
c = 3;
k = 1;%ͶӰֱ��ϵ��
%%��һ��
w1 = [0.42 -0.087 0.58; -0.2 -3.3 -3.4; 1.3 -0.32 1.7; 0.39 0.71 0.23; -1.6 -5.3 -0.15; 
    -0.029 0.89 -4.7; -0.23 1.9 2.2; 0.27 -0.3 -0.87; -1.9 0.76 -2.1; 0.87 -1.0 -2.6];
%%�ڶ���
w2 = [-0.4 0.58 0.089; -0.31 0.27 -0.04; 0.38 0.055 -0.035; -0.15 0.53 0.011; -0.35 0.47 0.034;
    0.17 0.69 0.1; -0.011 0.55 -0.18; -0.27 0.61 0.12; -0.065 0.49 0.0012; -0.12 0.054 -0.063];
%%������
w3 = [0.83 1.6 -0.014; 1.1 1.6 0.48; -0.44 -0.41 0.32; 0.047 -0.45 1.4; 0.28 0.35 3.1; 
    -0.39 -0.48 0.11; 0.34 -0.079 0.14; -0.3 -0.22 2.2; 1.1 1.2 -0.46; 0.18 -0.11 -0.49];
%%�����ֵ
m1 = mean(w1)';
m2 = mean(w2)';
m3 = mean(w3)';
m = (N * m1 + N * m2 + N * m3) / (3 * N);%%�����ֵ����
%%��������ɢ������
S1 = Intraclass_DM(w1, m1, N);
S2 = Intraclass_DM(w2, m2, N);
S3 = Intraclass_DM(w3, m3, N);

%%���w1��w2��w3���������ŷ���ʸ��w
Sw = S1 + S2 + S3;
Sb = Interclass_DM(m1, m2, m3, m, N, c);%�������ɢ������
S = inv(Sw) * Sb; 
[V, D] = eig(S);%%D�ĶԽ���Ԫ��������ֵ��V��������Ӧ����������
[D_sort, index] = sort(diag(D),'descend');
V_sort = V(:,index);
W1 = V_sort(:, 1); W2 = V_sort(:, 2);
W1 = W1 / norm(W1); W2 = W2 / norm(W2);%��λ��
W = [W1 W2];
y1 = W' * w1';%%�������w1��w2��w3���������ݵ��ڶ�ά�ӿռ�W�ϵ�ͶӰ
y2 = W' * w2';
y3 = W' * w3';
figure(1); WW1 = plot3(w1(:, 1), w1(:, 2), w1(:, 3), 'p');%%��w1,w2��w3����ά����ɢ��ͼ
hold on; grid on; WW2 = plot3(w2(:, 1), w2(:, 2), w2(:, 3), 'o');
WW3 = plot3(w3(:, 1), w3(:, 2), w3(:, 3), '*'); 
legend([WW1, WW2, WW3], 'w1����', 'w2����', 'w3����');
title('w1��w2��w3����ά����ɢ��ͼ');
figure(2); Y1 = plot(y1(1, :), y1(2, :), '^');%%����������ݵĶ�άɢ��ͼ
hold on; grid on; Y2 = plot(y2(1, :), y2(2, :), '.');
Y3 = plot(y3(1, :), y3(2, :), '+');
legend([Y1, Y2, Y3], 'w1ͶӰ��', 'w2ͶӰ��', 'w3ͶӰ��');
title('w1��w2��w3�������ӿռ��е�ͶӰ��(MDA)');

%%����ͶӰ��õ��������ά���ݵľ�ֵ������Э�������
miu1 = mean(y1')'; miu2 = mean(y2')'; miu3 = mean(y3')';
s1 = Cov(y1, miu1, N); s2 = Cov(y2, miu2, N); s3 = Cov(y3, miu3, N);
% s1 = cov(y1'); s2 = cov(y2'); s3 = cov(y3');%�����������������Э���������
%%������С�����ʱ�Ҷ˹��������ͶӰ��õ��������ά���ݼ��Ͻ��з���
p_w1 = 1/3; p_w2 = 1/3; p_w3 = 1/3;%%��Ҷ˹���������������
% [error_1,error_2,error_3,~,~,~]=Proj03_02_MADbayesclassify(y1',y2',y3',miu1', miu2', miu3', s1, s2, s3, p_w1, p_w2, p_w3);%��
[f_max1, pre_b1] = Bayes_cla(y1, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
[f_max2, pre_b2] = Bayes_cla(y2, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
[f_max3, pre_b3] = Bayes_cla(y3, miu1, miu2, miu3, s1, s2, s3, p_w1, p_w2, p_w3);
%%�����������ѵ��������ֵ�ĸ���
label1 = ones(N, 1); label2 = 2 * ones(N, 1); label3 = 3 * ones(N, 1);%��ǩ
error1 = length(find((pre_b1 - label1)~=0));%w1�������ĸ���
error2 = length(find((pre_b2 - label2)~=0));%w2�������ĸ���
error3 = length(find((pre_b3 - label3)~=0));%w3�������ĸ���
fprintf('�������ӿռ���(MDA)��\nʹ�ñ�Ҷ˹��������w1��w2��w3���������ݵ���з���\n');
fprintf('------------w1������: ��%d��-------------\n', pre_b1);
fprintf('w1���ݵ�ѵ������ֵ�Ϊ%d��\n\n', error1);
fprintf('------------w2������: ��%d��-------------\n', pre_b2);
fprintf('w2���ݵ�ѵ������ֵ�Ϊ%d��\n\n', error2);
fprintf('------------w3������: ��%d��-------------\n', pre_b3);
fprintf('w3���ݵ�ѵ������ֵ�Ϊ%d��\n\n', error3);

%% �Ա�ʵ��:�ڷ������ӿռ��У�ʹ�ñ�Ҷ˹����������w2��w3���ݵ�ѵ�����
v1 = [1.0 2.0 -1.5]'; v2 = [-1.0 0.5 -1.0]';
ww1 = v1 / norm(v1);%��λ��
ww2 = v2 / norm(v2);
%%���w2��w3���������ݵ���ʸ������w�ϵ�ͶӰ
W = [ww1 ww2];
yy1 = W' * w1';%%�������w1��w2��w3���������ݵ��ڶ�ά�ӿռ�W�ϵ�ͶӰ
yy2 = W' * w2';
yy3 = W' * w3';
figure(3); YY1 = plot(yy1(1, :), yy1(2, :), '^');%%��ǳ�ͶӰ��ĵ���ֱ���ϵ�λ�� 
hold on; grid on; YY2 = plot(yy2(1, :), yy2(2, :), '.');
YY3 = plot(yy3(1, :), yy3(2, :), '+');
legend([YY1, YY2, YY3], 'w1ͶӰ��', 'w2ͶӰ��', 'w3ͶӰ��');
title('w1��w2��w3�ڷ������ӿռ��е�ͶӰ��');

%%����ͶӰ��õ���������ά���ݵľ�ֵ�ͷ���
Miu1 = mean(yy1')'; Miu2 = mean(yy2')'; Miu3 = mean(yy3')';
ss1 = Cov(yy1, Miu1, N); ss2 = Cov(yy2, Miu2, N); ss3 = Cov(yy3, Miu3, N);
% ss1 = cov(yy1'); ss2 = cov(yy2'); ss3 = cov(yy3');%�����������������Э���������
%%������С�����ʱ�Ҷ˹��������ѵ�����������з���
p_w1= 1/3; p_w2 = 1/3; p_w3 = 1/3;%%��Ҷ˹���������������
[F_max1, Pre_b1] = Bayes_cla(yy1, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
[F_max2, Pre_b2] = Bayes_cla(yy2, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
[F_max3, Pre_b3] = Bayes_cla(yy3, Miu1, Miu2, Miu3, ss1, ss2, ss3, p_w1, p_w2, p_w3);
%%�����������ѵ��������ֵ�ĸ���
Error1 = length(find((Pre_b1 - label1)~=0));%w1�������ĸ���
Error2 = length(find((Pre_b2 - label2)~=0));%w1�������ĸ���
Error3 = length(find((Pre_b3 - label3)~=0));%w1�������ĸ���
fprintf('\n�ڷ������ӿռ��У�\nʹ�ñ�Ҷ˹��������w1��w2��w3���������ݵ���з���\n');
fprintf('------------w1������: ��%d��-------------\n', Pre_b1);
fprintf('w1���ݵ�ѵ������ֵ�Ϊ%d��\n\n', Error1);
fprintf('------------w2������: ��%d��-------------\n', Pre_b2);
fprintf('w2���ݵ�ѵ������ֵ�Ϊ%d��\n\n', Error2);
fprintf('------------w3������: ��%d��-------------\n', Pre_b3);
fprintf('w3���ݵ�ѵ������ֵ�Ϊ%d��\n\n', Error3);

%% ---------------------�Ӻ���-------------------------- %%
function S = Intraclass_DM(x, m, N) %%��������ɢ������xΪ����mΪ������NΪ������Ŀ��������
S = zeros(size(m, 1));
for i = 1: N
    A = (x(i, :)' - m) * (x(i, :)' - m)';
    S = A + S; 
end
end

function Sb = Interclass_DM(m1, m2, m3, m, N, c) %%�������ɢ������xΪ����mΪ������cΪ�����������
mm = [m1 m2 m3];
Sb = zeros(size(c, 1));
for i = 1: c
    A = N .* (mm(:, i) - m) * (mm(:, i) - m)';
    Sb = A + Sb; 
end
end

%%Э���������㺯��
function S = Cov(x, m, N) %%xΪ����mΪ������NΪ������Ŀ��������
S = zeros(size(m, 1));
for i = 1: N
    A = (1 / N) .* ((x(:, i)' - m) * (x(:, i)' - m)');
    S = A + S; 
end
end

%%���һ������w�ϵ�һά��Ҷ˹������
function [f_max, pre_b] = Bayes_cla(x, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3)
N = size(x, 2);
p1 = zeros(N, 1); p2 = zeros(N, 1); p3 = zeros(N, 1);
g1 = zeros(N, 1); g2 = zeros(N, 1); g3 = zeros(N, 1);
f_max = zeros(N, 1); pre_b = zeros(N, 1);
for i = 1 : N
    p1(i) = mvnpdf(x(:, i)', m1', S1); %��������
    p2(i) = mvnpdf(x(:, i)', m2', S2);
    p3(i) = mvnpdf(x(:, i)', m3', S3);
    g1(i) = p1(i) .* p_w1; %����������
    g2(i) = p2(i) .* p_w2;
    g3(i) = p3(i) .* p_w3;
    [f_max(i), pre_b(i)] = max([g1(i); g2(i); g3(i)]);
end
end
