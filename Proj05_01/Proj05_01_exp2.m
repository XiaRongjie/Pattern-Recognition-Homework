%%----------------Proj05-01����֪���㷨--------------%%
%%----------Proj05-01-exp2��������ԣ���ɳ��㷨-----------%%
clear; clc;
%%��һ��
w1 = [0.1 1.1; 6.8 7.1; -3.5 -4.1; 2.0 2.7; 4.1 2.8; 
    3.1 5.0; -0.8 -1.3; 0.9 1.2; 5.0 6.4; 3.9 4.0];
%%�ڶ���
w2 = [7.1 4.2; -1.4 -4.3; 4.5 0.0; 6.3 1.6; 4.2 1.9;
    1.4 -3.2; 2.4 -4.0; 2.5 -6.1; 8.4 3.7; 4.1 -2.2];
%%������
w3 = [-3.0 -2.9; 0.54 8.7; 2.9 2.1; -0.1 5.2; -4.0 2.2; 
    -1.3 3.7; -3.4 6.2; -4.1 3.4; -5.1 1.6; 1.9 5.1];
%%������
w4 = [-2.0 -8.4; -8.9 0.2; -4.2 -7.7; -8.5 -3.2; -6.7 -4.0; 
    -0.5 -9.2; -5.3 -6.7; -8.7 -6.4; -7.1 -9.7; -8.0 -6.3];
%%�������ݹ淶��
Y_w1 = [ones(size(w1, 1), 1), w1];
Y_w2 = [ones(size(w2, 1), 1), w2];
Y_w3 = [ones(size(w3, 1), 1), w3];
Y_w4 = [ones(size(w4, 1), 1), w4];

%% ��w1��w3��ѵ��������ʵ��
figure(1); title('ѵ������w1��w3�ķֲ�');%��ʾѵ������w1��w2�ķֲ�
plot(w1(:, 1), w1(:, 2), 'o');
hold on; grid on; plot(w3(:, 1), w3(:, 2), '+');
xlabel('x_1'); ylabel('x_2');
legend('w1', 'w3');
%��ʼ��
a = [0 0 0]'; eta = [0.01, 0.02]; b = [0.1 0.5]; 
Y = [Y_w1; -Y_w3]';%���࣬һ���Ϊ+1����һ��Ϊ-1

figure(2); hold on; title('��ѵ������w1��w3��J_r(a)��ֵ����ѵ���غ���k�ĺ�������');
for i = 1: size(eta, 2)
    for j = 1: size(b, 2)
        [ak, Jr_a, k] = batch_margin_relaxation(Y, a, eta(i), b(j));
        if b(j) == b(1) && eta(i) == eta(1)
            plot(0: k-2, Jr_a, '-'); hold on;
        end
        if b(j) == b(1) && eta(i) == eta(2)
            plot(0: k-2, Jr_a, '.'); hold on;
        end
        if b(j) == b(2) && eta(i) == eta(1)
            plot(0: k-2, Jr_a, '-o');
        end
        if b(j) == b(2) && eta(i) == eta(2)
            plot(0: k-2, Jr_a, '-^');
        end
    end
end
xlabel('k'); ylabel('J_r(a)');
legend('b = 0.1, \eta = 0.01', 'b = 0.5, \eta = 0.01', 'b = 0.1, \eta = 0.02', 'b = 0.5, \eta = 0.02');

%% ����
figure(3); title('ѵ������w1��w3�ķ���');%��ʾѵ������w1��w2�ķֲ�
plot(w1(:, 1), w1(:, 2), 'o');
hold on; grid on; plot(w3(:, 1), w3(:, 2), '+');
xmin = min(min(w1(:,1)),min(w3(:,1)));
xmax = max(max(w1(:,1)),max(w3(:,1)));
xindex = xmin-1: (xmax-xmin)/100: xmax+1;
yindex = -ak(2)*xindex/ak(3) - ak(1)/ak(3);
plot(xindex, yindex);
xlabel('x_1'); ylabel('x_2');
legend('w1', 'w3', '������');

%% �Ӻ���
function [a, Jr_a, k] = batch_margin_relaxation(Y, a, eta, b)
%������ԣ���ɳ��㷨
%���룺YΪ�������ݣ�aΪ��ʼ����Ȩֵ������learning_rateΪѧϰ�ʣ�thetaΪ��ֵ
%�����aΪȨֵ������Jp_aΪ׼������kΪ����ʱ�ĵ�������
	k = 0;
    Jr_a = [];%׼����
    while(1)
        k = k + 1;
        Yk = Y(:, a'*Y <= b);
        if isempty(Yk) || k > 300  %%�ж��������Ƿ�Ϊ��
            break;
        end
        sum1 = [0 0 0]';
        sum2 = 0;
        for i = 1: size(Yk, 2)
            sum1 = sum1 + (b - a'*Yk(:, i)) / (norm(Yk(:, i))^2) * Yk(:, i);
            sum2 = sum2 + (a'*Yk(:, i) - b)^2 / (norm(Yk(:, i))^2);
        end
        a = a + eta * sum1;%����Ȩֵ
        Jr_a = [Jr_a, 0.5 * sum2];%����׼����
    end
end
