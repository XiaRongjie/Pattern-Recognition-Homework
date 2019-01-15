function [Wji, Wjb, Wkj, Wkb, d_wkj, d_wji, Z1] = BP_train(eta, Wji, Wjb, Wkj, Wkb, X, T)
    %%���Ӻ�������BP����ѧϰ
    %���룺ѧϰ��eta����ʼȨֵWji, Wjb, Wkj, Wkb���������������X��������ʦ�ź�T
    %���������ȨֵWji, Wjb, Wkj, Wkb��Ȩֵ�仯��d_wkj, d_wji��������������Z1
    %%ǰ�����
    Y0 = Wji * X' + Wjb;
    Y1 = 1 ./ (1 + exp(-Y0));
    Z0 = Wkj * Y1 + Wkb;
    Z1 = 1 ./ (1 + exp(-Z0));
    %�����ݶ�
    dZ = (T - Z1) .* (Z1 .* (1 - Z1));
    d_wkj = eta * dZ * Y1';
    d_wji = eta * (Wkj' * dZ .* Y1 .* (1 - Y1)) * X;
    %����Ȩֵ
    Wkj = Wkj + d_wkj;  
    Wji = Wji + d_wji;
end
