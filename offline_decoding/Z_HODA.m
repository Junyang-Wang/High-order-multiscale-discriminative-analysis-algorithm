function [G,U]=Z_HODA(Data,R,Flag)
Dimensions= ndims(Data);
switch Dimensions
    case 5
        [Nc,Nt,Nf,Ns,Cls] = size(Data);
        X = tensor(Data(:,:,:,:));
        X_bk = bsxfun(@minus,Data,mean(Data,4)); %导联*时间*频率*试次*类别
        Gbl_M = mean(Data(:,:,:,:),4); % 总体样本的均值 ：导联*时间*频率
        Cls_M = squeeze(mean(Data,4)); % 各类样本的均值 ：导联*时间*频率*类别
        Xc_v = (Cls_M-repmat(Gbl_M,1,1,1,Cls))*sqrt(Ns); % 各类样本均值 -总体样本均值
        X_v = tensor(Xc_v);
        X_b = tensor(X_bk(:,:,:,:)); % Tensor 导联*时间*频率*（试次*类别）各样本mode-（N+1）拼接
        if isempty(R)
           R = [Nc,Nt,Nf];
        end
    case 4
        [Nc,Nt,Ns,Cls] = size(Data);
        X = tensor(Data(:,:,:));
        X_bk = bsxfun(@minus,Data,mean(Data,3)); %导联*时间*频率*试次*类别
        Gbl_M = mean(Data(:,:,:),3); % 总体样本的均值 ：导联*时间*频率
        Cls_M = squeeze(mean(Data,3)); % 各类样本的均值 ：导联*时间*频率*类别
        Xc_v = (Cls_M-repmat(Gbl_M,1,1,Cls))*sqrt(Ns); % 各类样本均值 -总体样本均值
        X_v = tensor(Xc_v);
        X_b = tensor(X_bk(:,:,:)); % Tensor 导联*时间*频率*（试次*类别）各样本mode-（N+1）拼接
        if isempty(R)
           R = [Nc,Nt];
        end
end
%% 
M = ndims(X_b)-1;
for n=1:M
    T_int{n} = nvecs(X_b,n,R(n));
end
U = T_int;
if Flag=='FixTime'
    U{1,2} = eye(R(2),R(2));
    N_List = setdiff(1:M,[2]);
else
    N_List = 1:M;
end
%%
for n = N_List
    Vld_M = setdiff(1:M,n);
    Z_b = ttm(X_b, U(1,Vld_M), Vld_M, 't'); % 所有样本拼接矩阵X_b 在非n模式下乘以 U' 得到Z_b
    Sw_n = ttt(Z_b,Z_b,[Vld_M M+1],[Vld_M M+1]); % 投影后的Z_b 求内积
    Z_v = ttm(X_v, U(1,Vld_M), Vld_M, 't'); % 各类样本均值-总体样本均值 拼接矩阵 X_b 在非n模式下乘以 U' 得到Z_v
    Sb_n = ttt(Z_v,Z_v,[Vld_M M+1],[Vld_M M+1]); % 投影后的Z_v 求内积
    St_n = Sw_n+Sb_n;
    [V_raw, D_raw] = eigs(double(St_n)\double(Sb_n),R(n));
    eigvalue=diag(D_raw);
    [D,index]=sort(eigvalue(:,1),1,'descend');
    U{1,n} = V_raw(:,index);
    P{1,n} = D/sum(D);
    PN{1,n} = cell2mat(arrayfun(@(x)(sum(P{1,n}(1:x,1),1)),1:R(n),'UniformOutput',false));
end
G = ttm(X,U,1:M, 't');
