%%----------------Proj02-02����С���Ͼ��������--------------%%
%% --�����������ݾ�ֵ��Э������󣬲��ֱ�ʹ����С���Ͼ������������Сŷʽ�������������С�����ʱ�Ҷ˹�������Դ�������������з���--%%
clear; clc;
%%����ÿһ�������ľ�ֵʸ����Э�������
N = 10;
c = 3;
%%��һ��
w1 = [-5.01 -8.12 -3.68; -5.43 -3.48 -3.54; 1.08 -5.52 1.66; 0.86 -3.78 -4.11; -2.67 0.63 7.39;
    4.94 3.29 2.08; -2.51 2.09 -2.59; -2.25 -2.13 -6.94; 5.56 2.86 -2.26; 1.03 -3.33 4.33];
%%�ڶ���
w2 = [-0.91 -0.18 -0.05; 1.30 -.206 -3.53; -7.75 -4.54 -0.95; -5.47 0.50 3.92; 6.14 5.72 -4.85;
    3.60 1.26 4.36; 5.37 -4.63 -3.65; 7.18 1.46 -6.66; -7.39 1.17 6.30; -7.50 -6.32 -0.31];
%%������
w3 = [5.35 2.26 8.13; 5.12 3.22 -2.66; -1.34 -5.31 -9.87; 4.48 3.42 5.19; 7.11 2.39 9.21; 
    7.17 4.33 -0.98; 5.75 3.97 6.65; 0.77 0.27 2.41; 0.90 -0.43 -8.71; 3.52 -0.36 6.43];
%%�����ֵ
m1 = mean(w1)';
m2 = mean(w2)';
m3 = mean(w3)';
%%����Э�������
S1 = Cov(w1, m1, N);
S2 = Cov(w2, m2, N);
S3 = Cov(w3, m3, N); 
%%��ͼ
figure(1); draw(m1, S1);
figure(2); draw(m2, S2);
figure(3); draw(m3, S3);
%%������С���Ͼ���������Բ�����������з���
test1 = [1 2 1]; test2 = [5 3 2]; test3 = [0 0 0]; test4 = [1 0 0]; 
[g_max1, pre_m1] = Ma_cla(test1, m1, m2, m3, S1, S2, S3, c);
[g_max2, pre_m2] = Ma_cla(test2, m1, m2, m3, S1, S2, S3, c);
[g_max3, pre_m3] = Ma_cla(test3, m1, m2, m3, S1, S2, S3, c);
[g_max4, pre_m4] = Ma_cla(test4, m1, m2, m3, S1, S2, S3, c);
fprintf('--������С���Ͼ���������Բ�����������з���--\n');
fprintf('--------------���Ե�1: ��%d��---------------\n', pre_m1);
fprintf('--------------���Ե�2: ��%d��---------------\n', pre_m2);
fprintf('--------------���Ե�3: ��%d��---------------\n', pre_m3);
fprintf('--------------���Ե�4: ��%d��---------------\n\n', pre_m4);
%%������Сŷʽ����������Բ�����������з���
[h_max1, pre_e1] = Eu_cla(test1, m1, m2, m3, c);
[h_max2, pre_e2] = Eu_cla(test2, m1, m2, m3, c);
[h_max3, pre_e3] = Eu_cla(test3, m1, m2, m3, c);
[h_max4, pre_e4] = Eu_cla(test4, m1, m2, m3, c);
fprintf('--������Сŷʽ����������Բ�����������з���--\n');
fprintf('--------------���Ե�1: ��%d��---------------\n', pre_e1);
fprintf('--------------���Ե�2: ��%d��---------------\n', pre_e2);
fprintf('--------------���Ե�3: ��%d��---------------\n', pre_e3);
fprintf('--------------���Ե�4: ��%d��---------------\n\n', pre_e4);
%%������С�����ʱ�Ҷ˹�������Բ�����������з���
p_w1 = 1/3; p_w2 = 1/3; p_w3 = 1/3; %�������
[f_max1, pre_b1] = Bayes_cla(test1, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3);
[f_max2, pre_b2] = Bayes_cla(test2, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3);
[f_max3, pre_b3] = Bayes_cla(test3, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3);
[f_max4, pre_b4] = Bayes_cla(test4, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3);
fprintf('--������С�����ʱ�Ҷ˹�������Բ�����������з���--\n');
fprintf('--------------���Ե�1: ��%d��---------------\n', pre_b1);
fprintf('--------------���Ե�2: ��%d��---------------\n', pre_b2);
fprintf('--------------���Ե�3: ��%d��---------------\n', pre_b3);
fprintf('--------------���Ե�4: ��%d��---------------\n', pre_b4);

%% -----------���������ɱ������--------------- %%
data_row1 = [pre_m1, pre_m2, pre_m3, pre_m4];
data_row2 = [pre_e1, pre_e2, pre_e3, pre_e4];
data_row3 = [pre_b1, pre_b2, pre_b3, pre_b4];
data=[data_row1; data_row2; data_row3];
%%���ɱ���������ƣ�m��n��
str1='����������';str2='����';
m=3; n=4;
column_name = strcat(str1,num2str((1:n)'));
row_name = {'���Ϸ�����', 'ŷʽ������', '��Ҷ˹������'}';
%%�����ͼ
set(figure(4),'position',[200 200 450 150]);
uitable(gcf,'Data',data,'Position',[20 20 400 100],'Columnname',column_name,'Rowname',row_name);

%% ---------------------�Ӻ���-------------------------- %%
%%Э���������㺯��
function S = Cov(x, m, N) %%xΪ����mΪ������NΪ������Ŀ��������
S = zeros(size(m, 1));
for i = 1: N
    A = (1 / N) .* ((x(i, :)' - m) * (x(i, :)' - m)');
    S = A + S; 
end
end

%%�������Ͼ���
function D = Ma_dis(x, m, S)
    D = sqrt((x' - m)' / S * (x' - m));
end

%%���Ͼ���ֲ�ͼ
function draw(m, S)
    t = 25;
    x = m(1) - t : 1 : m(1) + t; 
    y = m(2) - t : 1 : m(2) + t;
    z = m(3) - t : 1 : m(3) + t;
    [X, Y, Z]  = meshgrid(x, y, z);
    %color = ['r', 'g', 'b'];
    for D = 1: 3
        point = [];
        for j = 1: size(X(:))
            if abs(Ma_dis([X(j), Y(j), Z(j)], m, S) - D) < 0.1
                point = [point, [X(j); Y(j); Z(j)]];
            end
        end
        scatter3(point(1, :), point(2, :), point(3, :));
        hold on;
        
    end
    legend('���Ͼ���D=1��������', '���Ͼ���D=2��������', '���Ͼ���D=3��������');
end

%%��С���Ͼ��������
function [g_max, pre_m] = Ma_cla(x, m1, m2, m3, S1, S2, S3, c)
m = [m1, m2, m3];
S = [S1, S2, S3];
g = zeros(c, 1);
for i = 1: c
    g(i) = - Ma_dis(x, m(:, i), S(:, 3*i-2:3*i));
end
[g_max, pre_m] = max(g);
end

%%����ŷʽ����
function E = Eu_dis(x, m)
    E = sqrt(sum((x' - m) .* (x' - m)));
end

%%��Сŷʽ���������
function [h_max, pre_e] = Eu_cla(x, m1, m2, m3, c)
m = [m1, m2, m3];
g = zeros(c, 1);
for i = 1: c
    g(i) = - Eu_dis(x, m(:, i));
end
[h_max, pre_e] = max(g);
end

%%��С�����ʱ�Ҷ˹������
function [f_max, pre_b] = Bayes_cla(x, m1, m2, m3, S1, S2, S3, p_w1, p_w2, p_w3)
p1 = mvnpdf(x, m1', S1); %��������
p2 = mvnpdf(x, m2', S2);
p3 = mvnpdf(x, m3', S3);
g1 = p1 .* p_w1; %����������
g2 = p2 .* p_w2;
g3 = p3 .* p_w3;
[f_max, pre_b] = max([g1; g2; g3]);
end


