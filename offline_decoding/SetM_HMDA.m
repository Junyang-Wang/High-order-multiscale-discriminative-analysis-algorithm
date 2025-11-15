function [EMdl]=SetM_HMDA(Data,R,Flag)
%% 数据格式：5D:导联*时间*频率*试次*类别  4D：导联*时间*试次*类别
% tic
[Nc,Nt,Nf,Ns,Cls] = size(Data);
EMdl(1).TrnDt_Crr=[];
%% for 3D part 
[~,U_3D] = Z_HODA(Data,[],Flag);
for n = 1:3
    EMdl(1).U_3D{1,n} = U_3D{1,n}(:,1:R(n));
end
G_3D = ttm(tensor(Data(:,:,:,:)),EMdl(1,1).U_3D,1:3, 't');
%%
Trn_G_3D = reshape(double(G_3D),[R(1),R(2),R(3),R(4)/Cls,Cls]);
Trn_G_3D = bsxfun(@minus,Trn_G_3D,mean(Trn_G_3D,2)); 
G_3D0 = permute(Trn_G_3D,[4 5 1 2 3]);
Cat_G_3D = G_3D0(:,:,:,:);
Tmpt_3D = squeeze(mean(Trn_G_3D,4));
Tmpt_3 = permute(Tmpt_3D,[4 1 2 3]);
EMdl(1).Tmpt_3D = Tmpt_3(:,:,:);
% low freq-half
Data_LF = squeeze(Data(:,:,1:6,:,:));
[~,U_3D_LF] = Z_HODA(Data_LF,[],Flag);
r = [Nc,Nt,1];
for n = 1:3
    EMdl(1).U_3D_LF{1,n} = U_3D_LF{1,n}(:,1:r(n));
end
G_3D_LF = ttm(tensor(Data_LF(:,:,:,:)),EMdl(1,1).U_3D_LF,1:3, 't');
Trn_G_3D_LF = reshape(double(G_3D_LF),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_3D_LF = bsxfun(@minus,Trn_G_3D_LF,mean(Trn_G_3D_LF,2)); 
EMdl(1).Tmpt_LF = squeeze(mean(Trn_G_3D_LF,4));
% high freq-half
Data_HF = squeeze(Data(:,:,7:12,:,:));
[~,U_3D_HF] = Z_HODA(Data_HF,[],Flag);
for n = 1:3
    EMdl(1).U_3D_HF{1,n} = U_3D_HF{1,n}(:,1:r(n));
end
G_3D_HF = ttm(tensor(Data_HF(:,:,:,:)),EMdl(1,1).U_3D_HF,1:3, 't');
Trn_G_3D_HF = reshape(double(G_3D_HF),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_3D_HF = bsxfun(@minus,Trn_G_3D_HF,mean(Trn_G_3D_HF,2)); 
EMdl(1).Tmpt_HF = squeeze(mean(Trn_G_3D_HF,4));
% low freq-a quarter 13
Data_LF_13 = squeeze(Data(:,:,1:3,:,:));
[~,U_3D_LF_13] = Z_HODA(Data_LF_13,[],Flag);
for n = 1:3
    EMdl(1).U_LF_13{1,n} = U_3D_LF_13{1,n}(:,1:r(n));
end
G_3D_LF_13 = ttm(tensor(Data_LF_13(:,:,:,:)),EMdl(1,1).U_LF_13,1:3, 't');
Trn_G_LF_13 = reshape(double(G_3D_LF_13),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_LF_13 = bsxfun(@minus,Trn_G_LF_13,mean(Trn_G_LF_13,2)); 
EMdl(1).Tmpt_LF_13 = squeeze(mean(Trn_G_LF_13,4));
% low freq-a quarter 46
Data_LF_46 = squeeze(Data(:,:,4:6,:,:));
[~,U_3D_LF_46] = Z_HODA(Data_LF_46,[],Flag);
for n = 1:3
    EMdl(1).U_LF_46{1,n} = U_3D_LF_46{1,n}(:,1:r(n));
end
G_3D_LF_46 = ttm(tensor(Data_LF_46(:,:,:,:)),EMdl(1,1).U_LF_46,1:3, 't');
Trn_G_LF_46 = reshape(double(G_3D_LF_46),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_LF_46 = bsxfun(@minus,Trn_G_LF_46,mean(Trn_G_LF_46,2)); 
EMdl(1).Tmpt_LF_46 = squeeze(mean(Trn_G_LF_46,4));
% high freq-a quarter 79
Data_HF_79 = squeeze(Data(:,:,7:9,:,:));
[~,U_3D_HF_79] = Z_HODA(Data_HF_79,[],Flag);
for n = 1:3
    EMdl(1).U_HF_79{1,n} = U_3D_HF_79{1,n}(:,1:r(n));
end
G_3D_HF_79 = ttm(tensor(Data_HF_79(:,:,:,:)),EMdl(1,1).U_HF_79,1:3, 't');
Trn_G_HF_79 = reshape(double(G_3D_HF_79),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_HF_79 = bsxfun(@minus,Trn_G_HF_79,mean(Trn_G_HF_79,2)); 
EMdl(1).Tmpt_HF_79 = squeeze(mean(Trn_G_HF_79,4));
% high freq-a quarter 10 12
Data_HF_HF = squeeze(Data(:,:,10:12,:,:));
% [Dim1,Dim2,Dim3,Dim4,Dim5] = size(Data_HF_HF);
[~,U_3D_HF_HF] = Z_HODA(Data_HF_HF,[],Flag);
for n = 1:3
    EMdl(1).U_HF_HF{1,n} = U_3D_HF_HF{1,n}(:,1:r(n));
end
G_3D_HF_HF = ttm(tensor(Data_HF_HF(:,:,:,:)),EMdl(1,1).U_HF_HF,1:3, 't');
Trn_G_HF_HF = reshape(double(G_3D_HF_HF),[r(1),r(2),r(3),Ns,Cls]);
Trn_G_HF_HF = bsxfun(@minus,Trn_G_HF_HF,mean(Trn_G_HF_HF,2)); 
EMdl(1).Tmpt_HF_HF = squeeze(mean(Trn_G_HF_HF,4));
% 
% tic
for cls_w=1:Cls
    for cls=1:Cls
    Crr_3D(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Cat_G_3D(x,cls,:,:))),real(squeeze(EMdl(1).Tmpt_3D(cls_w,:,:))))),1:Ns,'UniformOutput',false));
    Crr_LF(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_3D_LF(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_LF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    Crr_HF(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_3D_HF(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_HF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    Crr_LF_13(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_LF_13(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_LF_13(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    Crr_LF_46(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_LF_46(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_LF_46(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    Crr_HF_79(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_HF_79(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_HF_79(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    Crr_HF_HF(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Trn_G_HF_HF(:,:,:,x,cls))),real(squeeze(EMdl(1).Tmpt_HF_HF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    end
end
% toc
EMdl(1).TrnDt_Crr=cat(2,Crr_3D(:,:)',Crr_LF(:,:)',Crr_HF(:,:)',Crr_LF_13(:,:)',Crr_LF_46(:,:)',Crr_HF_79(:,:)',Crr_HF_HF(:,:)');
%% for 2D part
CmbW=[];
% tic
for bk = 1:Nf
    Data_2D = squeeze(Data(:,:,bk,:,:));
    Data_2D = bsxfun(@minus,Data_2D,mean(Data_2D,2));
    [~,U_2D] = Z_HODA(Data_2D,R(1,[1 2]),Flag);
    CmbW(:,:,bk) = U_2D{1,1}';
    M_2Data = squeeze(mean(Data_2D,3));
    EMdl(1,bk).Tmpt = reshape(cell2mat(arrayfun(@(x)(U_2D{1,1}'*M_2Data(:,:,x)),1:Cls,'UniformOutput',false)),size(CmbW,1),size(M_2Data,2),Cls);
    for cls_w=1:Cls
        for cls=1:Cls
            Trn_Crr(cls_w,:,cls)=cell2mat(arrayfun(@(x)(corr2(real(U_2D{1,1}'*squeeze(Data_2D(:,:,x,cls))),real(squeeze(EMdl(1,bk).Tmpt(:,:,cls_w))))),1:Ns,'UniformOutput',false));
        end
    end
    EMdl(1).TrnDt_Crr=cat(2,EMdl(1).TrnDt_Crr,Trn_Crr(:,:)');
end
% toc
EMdl(1,1).CmbW_S = blkdiag(CmbW(:,:,1),CmbW(:,:,2),CmbW(:,:,3),CmbW(:,:,4),CmbW(:,:,5),CmbW(:,:,6),CmbW(:,:,7),CmbW(:,:,8),CmbW(:,:,9),CmbW(:,:,10),CmbW(:,:,11),CmbW(:,:,12));  
%%
Trnbl_Raw = repmat(1:Cls,Ns,1);
TrnLbl = Trnbl_Raw(:);
Group=nchoosek(1:Cls,2);
EMdl(1,1).Cls=Cls;
for grp=1:size(Group,1)
    Idex=cat(1,find(TrnLbl==Group(grp,1)),find(TrnLbl==Group(grp,2)));
    TrnLbl0=TrnLbl(Idex,1);
    Trn_Crr=EMdl(1).TrnDt_Crr(Idex,:);
    int_model = false(size(Trn_Crr,2),1);
    int_model(randperm(size(Trn_Crr,2),fix(size(Trn_Crr,2)/2)),1) = true;
    [b,se,pval,inmodel,stats,nextstep,history]=stepwisefit(Trn_Crr,TrnLbl0,'InModel',int_model,'penter',0.1,'premove',0.15,'display','off','maxiter',100);
    EMdl(grp,1).index =find(inmodel==1);
    EMdl(grp,1).CrSVM = fitcsvm(Trn_Crr(:,EMdl(grp,1).index),TrnLbl0);
end
% toc
end