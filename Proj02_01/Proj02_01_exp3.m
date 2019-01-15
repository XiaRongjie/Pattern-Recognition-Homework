%%----------------Proj02-01����С�����ʱ�Ҷ˹������--------------%%
%%----------------Proj02-01-exp3----------------%%
%%Э������󲻱䣬�����ֵ�����ֱ��Ϊm1 = (1, 3)T��m2 = (4, 0)T�����½���ʵ�飻
clear; clc;
m1 = [1; 3]; m2 = [4; 0]; %��ֵ
S1 = [1.5 0; 0 1]; S2 = [1 0.5; 0.5 2]; %Э�������
n = 100; %�����������Ϊ100
P1 = mvnrnd(m1, S1, n); %��һ������
P2 = mvnrnd(m2, S2, n); %�ڶ�������
subplot(1, 2, 1);
s1 = scatter(P1(:, 1), P1(:, 2), '.');
hold on; s2 = scatter(P2(:, 1), P2(:, 2), 'v');
title('���������Ķ�άɢ��ͼ');
legend([s1 s2], '��һ��������', '�ڶ���������');

P = [P1; P2]; %��������
p1 = mvnpdf(P, m1', S1); %��������
p2 = mvnpdf(P, m2', S2);
p_w1 = 0.5; p_w2 = 0.5; %�������
g1 = p1 .* p_w1; %����������
g2 = p2 .* p_w2;

%%��������������С�����ʱ�Ҷ˹������������������%%
g = g1 - g2; 
D = zeros(size(P, 1), 1);
D(find(g > 0)) = 1;
D(find(g < 0)) = 2;
D(find(g == 0)) = inf;
%%����������������������������������������������%%

label = [ones(size(P1, 1), 1); 2 * ones(size(P2, 1), 1)]; %200�������ı�ǩ
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
acc = sprintf('��ȷ����ٷֱ� = %f', accuracy);
st = ['������ķ�����:', string(acc)];
title(st);
legend([c w], '��ȷ����������', '�������������');
if ~isempty(unsure)
    u = scatter(unsure(:, 1), unsure(:, 2), '*');
    legend([c w u], '��ȷ����������', '�������������', '�������������');
end
%%
%%�ı�������ʣ����ñ�Ҷ˹����������ͬ�����������·���
p_w11 = 0.4; p_w22 = 0.6; %�������
g11 = p1 .* p_w11; %����������
g22 = p2 .* p_w22;

%%��������������С�����ʱ�Ҷ˹������������������%%
gg = g11 - g22; 
DD = zeros(size(P, 1), 1);
DD(find(gg > 0)) = 1;
DD(find(gg < 0)) = 2;
DD(find(gg == 0)) = inf;
%%����������������������������������������������%%

label = [ones(size(P1, 1), 1); 2 * ones(size(P2, 1), 1)]; %200�������ı�ǩ
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
acc1 = sprintf('��ȷ����ٷֱ� = %f', accuracy1);
st1 = ['������ķ�����:', string(acc1)];
title(st1);
legend([cc ww], '��ȷ����������', '�������������');
if ~isempty(unsure1)
    uu = scatter(unsure1(:, 1), unsure1(:, 2), '*');
    legend([cc ww uu], '��ȷ����������', '�������������', '�������������');
end
