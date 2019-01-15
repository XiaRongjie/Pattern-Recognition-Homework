%%----------------Proj03-01������������PCA--------------%%
%%--------------------Proj03-01-exp2-------------------%%
clc; clear;
N = 20;%��������
miu = [10; 15; 15];%��ֵ 
sigma = [90, 2.5, 1.2; 2.5, 35, 0.2; 1.2, 0.2, 0.02];%Э�������
X = mvnrnd(miu, sigma, N);%����N����˹�ֲ��Ķ�ά����ʸ��
figure(1); plot3(X(:, 1), X(:, 2), X(:, 3), 'o'); title('��������X����άɢ��ͼ');
m = mean(X)';%��������X�ľ�ֵ����
mm = repmat(m, 1, N);%%�þ���ķ������㣬repmat�Ǹ��ƺ�ƽ�̾���
S = (X' - mm) * (X' - mm)';
S1 = (N - 1) * cov(X);
[V, D] = eig(S1);%D�ĶԽ���Ԫ��������ֵ��V��������Ӧ����������
[D_sort, index] = sort(diag(D),'descend');
V_sort = V(:,index);
% Y = V * (X' - mm);
Y1 = V_sort(:, 1)' * (X' - mm);
Y2 = V_sort(:, 2)' * (X' - mm);
Y = [Y1; Y2];
figure(2); plot(Y(1, :), Y(2, :), '+'); title('��������Y�Ķ�άɢ��ͼ');
VV = inv(V);%���棬����ʹ��û������֮ǰ���������󣬲��ܵõ���ֱͶӰ���������������������ܵõ���ֱͶӰ����
W = VV(:, 1:2);
Z = W * Y + mm;
figure(3); XX = plot3(X(:, 1), X(:, 2), X(:, 3), 'o'); 
hold on; ZZ = plot3(Z(1, :), Z(2, :), Z(3, :), '*'); 
legend([XX, ZZ], '����X����', '����Z����');
title('����Z�ͼ���X����ά����ɢ��ͼ');
grid on;
for i = 1: N
    plot3([Z(1, i), X(i, 1)], [Z(2, i), X(i, 2)], [Z(3, i), X(i, 3)]);
end
grid on;
E = (X' - Z).^2;
Square_E = sum(sum(E));%�������е���Щ���ƽ��֮��
MeanSquare_E = (1/N) * Square_E;%�������ǵľ������
fprintf('���ƽ��֮�� = %f\n', Square_E);
fprintf('������� = %f\n', MeanSquare_E);