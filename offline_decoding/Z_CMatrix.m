function [CMatrix]=Z_CMatrix(TestLabel,PreCls)
%% TestLabel，PreCls：N*1向量，N为测试样本量
%% CMatrix：混淆矩阵，M*M矩阵，M为类别数

   CMatrix=zeros(size(unique(TestLabel),1),size(unique(TestLabel),1));
   for x=1:size(TestLabel,1) %size(Test_Data,1) 测试数据的个数 
        CMatrix(TestLabel(x),PreCls(x))=CMatrix(TestLabel(x),PreCls(x))+1;
   end 
end