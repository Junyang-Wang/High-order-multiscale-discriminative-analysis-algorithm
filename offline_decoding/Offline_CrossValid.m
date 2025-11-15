function [Acc] = Offline_CrossValid(data, Wvlt_DataFtr, Rc_Indx)
primay_dir = pwd;
TargetData = data;
Cls = size(TargetData,4);
RptNum = 5;
CrossNum = 5;
Rpts = 8;
for rpt=1:RptNum
    Indices = crossvalind('Kfold',fix(size(TargetData,3)/CrossNum)*CrossNum, CrossNum);
    tempIndex=1:fix(size(TargetData,3)/CrossNum)*CrossNum;
    for crossnum=1:CrossNum
        TrainIdex(:,crossnum,rpt)=tempIndex(Indices ~= crossnum);
        TestIdex(:,crossnum,rpt)=tempIndex(Indices == crossnum);
    end
end
TrainAryIdex = TrainIdex(:,:);
TestAryIdex = TestIdex(:,:);
for runs=1:size(TrainAryIdex,2)
    TrainData=TargetData(:,:,TrainAryIdex(:,runs),:);
    TestData_Raw=TargetData(:,:,TestAryIdex(:,runs),:);
    %%
    Trn_WvltFtr=Wvlt_DataFtr(:,:,TrainAryIdex(:,runs),:,:);
    Trn_Ftr=real(Trn_WvltFtr);
    Tst_WvltFtr=Wvlt_DataFtr(:,:,TestAryIdex(:,runs),:,:);
    Tst_Ftr_Raw=real(Tst_WvltFtr);
    %%
    TstLbl_Raw=repmat(1:Cls,size(TestData_Raw,3),1);
    TstLbl_Seq=TstLbl_Raw(:);
    blk_TstLbl=reshape(TstLbl_Seq,[Rpts,size(TstLbl_Seq,1)/Rpts]);
    TstLbl=blk_TstLbl(1,:)';
    %%
    Trnbl_Raw=repmat(1:Cls,size(TrainData,3),1);
    TrnLbl=Trnbl_Raw(:);
    %
    tmp=permute(Tst_Ftr_Raw,[5 1 2 3 4]);
    Test=reshape(tmp(:,:,:,:),[size(tmp,1) size(tmp,2) size(tmp,3) Rpts size(tmp,4)*size(tmp,5)/Rpts]);
    block_Test_WvltFtr=permute(Test,[2 3 4 5 1]);
    TstData_Seq=TestData_Raw(:,:,:);
    blk_TstData=permute(reshape(permute(TstData_Seq,[3 1 2]),[Rpts,size(TstLbl_Seq,1)/Rpts,size(TestData_Raw,1),size(TestData_Raw,2)]),[3,4,1,2]);
    %% train model
    Mdl_HMDA=SetM_HMDA(permute(Trn_Ftr,[1 2 5 3 4]),[7 26 7 size(TrnLbl,1)],'FixTime'); 
    %%
    Group=nchoosek(1:Cls,2);
    for rnd=1:Rpts 
        Tst_WavFtr(:,:,:,rnd)=Test_HMDA(squeeze(block_Test_WvltFtr(:,:,rnd,:,:)),Mdl_HMDA);
        rr = squeeze(sum(Tst_WavFtr(:,:,:,1:rnd),4));
        [~,PreCls_OVOSVM] = arrayfun(@(x)(max(rr(:,:,x),[],2)),1:size(Group,1),'UniformOutput', false);        
        OVO_Vote=[];OVO_Vote1=[];
        for grp=1:size(Group,1)
            OVO_Vote(grp,:)=cell2mat(arrayfun(@(x)(Group(grp,PreCls_OVOSVM{1,grp}(x,1))),1:size(TstLbl,1),'UniformOutput', false));
        end
        PreCls_VoteSVM=[];PreCls_VoteSVM1=[];
        for trl=1:size(TstLbl,1)
            tabl=tabulate(OVO_Vote(:,trl));
            [M,~]=max(tabl(:,2));

            Idx=find(tabl(:,2)==M);
            if length(Idx)==1
                idx=Idx;
            else
                Re_Vote=[];A=[];eql_Cls=[];Grp2=[];re_tabl=[];
                eql_Cls=tabl(Idx,1);
                Grp2=nchoosek(eql_Cls,2);
                for grp=1:size(Grp2,1)
                  A=Group-repmat(Grp2(grp,:),size(Group,1),1);
                  Row=find(all(A==0,2));
                  Re_Vote(grp,:)=OVO_Vote(Row,trl);
                end
                re_tabl=tabulate(Re_Vote);
                [~,r_idx]=max(re_tabl(:,2));
                idx=r_idx;
            end 
            PreCls_VoteSVM(:,trl)=tabl(idx,1);
        end
        Ac_HMDA(rnd,runs,:)=arrayfun(@(x)(TstLbl(x,1)==PreCls_VoteSVM(:,x)),1:size(TstLbl,1));
        CMatrix_HMDA(runs,rnd,:,:)=cell2mat(arrayfun(@(x)(Z_CMatrix(TstLbl,double(PreCls_VoteSVM)')),1,'UniformOutput', false));
        %%
    end
    disp([runs])
end
Acc.HMDA=squeeze(mean(mean(Ac_HMDA,2),3));
Acc.CMatrix_HMDA=squeeze(mean(CMatrix_HMDA,1));
end

function [dv]=Test_HMDA(TestData,Mdl)
% tic
Cls = Mdl(1,1).Cls;
Group = nchoosek(1:Cls,2);
[Nc,Nt,Ns,Nf]=size(TestData);
%% 3D
TstData_3D = tensor(permute(TestData,[1 2 4 3]));
G_3D = ttm(TstData_3D,Mdl(1,1).U_3D,1:3, 't');
Tst_3D = permute(double(G_3D),[1 2 4 3]);
Tst_3D = bsxfun(@minus,Tst_3D,mean(Tst_3D,2));
Tst_3D0 = permute(Tst_3D,[3 1 2 4]);
Cat_3D = Tst_3D0(:,:,:);
% low freq-half
TstData_LF =TstData_3D(:,:,1:6,:);
G_3D_LF = ttm(TstData_LF,Mdl(1,1).U_3D_LF,1:3, 't');
Tst_3D_LF = permute(double(G_3D_LF),[1 2 4 3]);
Tst_3D_LF = bsxfun(@minus,Tst_3D_LF,mean(Tst_3D_LF,2));
% high freq-half
TstData_HF =TstData_3D(:,:,7:12,:);
G_3D_HF = ttm(TstData_HF,Mdl(1,1).U_3D_HF,1:3, 't');
Tst_3D_HF = permute(double(G_3D_HF),[1 2 4 3]);
Tst_3D_HF = bsxfun(@minus,Tst_3D_HF,mean(Tst_3D_HF,2));
% low freq-a quarter 13
TstData_LF_13 =TstData_3D(:,:,1:3,:);
G_LF_13 = ttm(TstData_LF_13,Mdl(1,1).U_LF_13,1:3, 't');
Tst_LF_13 = permute(double(G_LF_13),[1 2 4 3]);
Tst_LF_13 = bsxfun(@minus,Tst_LF_13,mean(Tst_LF_13,2));
% low freq-a quarter 46
TstData_LF_46 =TstData_3D(:,:,4:6,:);
G_LF_46 = ttm(TstData_LF_46,Mdl(1,1).U_LF_46,1:3, 't');
Tst_LF_46 = permute(double(G_LF_46),[1 2 4 3]);
Tst_LF_46 = bsxfun(@minus,Tst_LF_46,mean(Tst_LF_46,2));
% high freq-a quarter 79
TstData_HF_79 =TstData_3D(:,:,7:9,:);
G_HF_79 = ttm(TstData_HF_79,Mdl(1,1).U_HF_79,1:3, 't');
Tst_HF_79 = permute(double(G_HF_79),[1 2 4 3]);
Tst_HF_79 = bsxfun(@minus,Tst_HF_79,mean(Tst_HF_79,2));
% high freq-a quarter HF
TstData_HF_HF =TstData_3D(:,:,10:12,:);
G_HF_HF = ttm(TstData_HF_HF,Mdl(1,1).U_HF_HF,1:3, 't');
Tst_HF_HF = permute(double(G_HF_HF),[1 2 4 3]);
Tst_HF_HF = bsxfun(@minus,Tst_HF_HF,mean(Tst_HF_HF,2));
% toc
% tic
Tst_Crr = [];
for cls_w=1:Cls
    temp1(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Cat_3D(x,:,:))),real(squeeze(Mdl(1).Tmpt_3D(cls_w,:,:))))),1:Ns,'UniformOutput',false));
    temp2(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_3D_LF(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_LF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    temp3(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_3D_HF(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_HF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    temp4(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_LF_13(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_LF_13(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    temp5(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_LF_46(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_LF_46(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    temp6(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_HF_79(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_HF_79(:,:,cls_w))))),1:Ns,'UniformOutput',false));
    temp7(cls_w,:)=cell2mat(arrayfun(@(x)(corr2(real(squeeze(Tst_HF_HF(:,:,x,:))),real(squeeze(Mdl(1).Tmpt_HF_HF(:,:,cls_w))))),1:Ns,'UniformOutput',false));
end
Tst_Crr = cat(1,temp1,temp2,temp3,temp4,temp5,temp6,temp7);
% toc
%% 2D
TstData=TestData-permute(repmat(squeeze(mean(TestData,2)),1,1,1,Nt),[1 4 2 3 ]);
temp_2D=[];temp=[];
for n=1:Ns
    A=blkdiag(TstData(:,:,n,1),TstData(:,:,n,2),TstData(:,:,n,3),TstData(:,:,n,4),TstData(:,:,n,5),TstData(:,:,n,6),TstData(:,:,n,7),TstData(:,:,n,8),...
        TstData(:,:,n,9),TstData(:,:,n,10),TstData(:,:,n,11),TstData(:,:,n,12));
    B=Mdl(1,1).CmbW_S*A;
    C=mat2cell(B,ones(1, Nf)*size(Mdl(1,1).CmbW_S,1)/Nf,ones(1,12)*Nt);
    for cls_w=1:Cls
        for bk=1:Nf
            temp(cls_w,bk)=corr2(real(C{bk,bk}),squeeze(Mdl(1,bk).Tmpt(:,:,cls_w)));
        end
    end
   temp_2D(:,n)=temp(:);
end
Tst_Crr=cat(1,Tst_Crr,temp_2D);
%%
for grp=1:size(Group,1)
    [~,dv(:,:,grp)]=predict(Mdl(grp,1).CrSVM,Tst_Crr(Mdl(grp,1).index,:)');
end
% toc
end
