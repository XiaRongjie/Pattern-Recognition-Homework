function [Wji, Wjb, Wkj, Wkb, d_wkj, d_wji, Z1] = BP_train(eta, Wji, Wjb, Wkj, Wkb, X, T)
    %%该子函数用于BP网络学习
    %输入：学习率eta，初始权值Wji, Wjb, Wkj, Wkb，输入层输入向量X，输出层教师信号T
    %输出：更新权值Wji, Wjb, Wkj, Wkb，权值变化量d_wkj, d_wji，输出层输出向量Z1
    %%前馈输出
    Y0 = Wji * X' + Wjb;
    Y1 = 1 ./ (1 + exp(-Y0));
    Z0 = Wkj * Y1 + Wkb;
    Z1 = 1 ./ (1 + exp(-Z0));
    %计算梯度
    dZ = (T - Z1) .* (Z1 .* (1 - Z1));
    d_wkj = eta * dZ * Y1';
    d_wji = eta * (Wkj' * dZ .* Y1 .* (1 - Y1)) * X;
    %更新权值
    Wkj = Wkj + d_wkj;  
    Wji = Wji + d_wji;
end
