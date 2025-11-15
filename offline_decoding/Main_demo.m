clc
close 
clear all
tic
Fs = 1000;
n_block = 5;
cns_21 = [44:64];
time = [1 0.25-0.055 0.055];
re_Fs = 100;
filtr_pars = 9;
bp_fltr = 20;
RdTrca_idx1 = 2;
RptNum = 1;
Rpts = 8;
CrossNum = 5;
wavename = 'cmor1-0.5';
F_Range = [0.1 20.1];
df = 1.6;
[data,Trn_Ftr] = data_prepocess(pwd,Fs,time,n_block,cns_21,re_Fs,bp_fltr,wavename,F_Range,df);
TrainData = data{1,1};
%%
[Nc,Nt,Ns,Cls] = size(TrainData);
Trnbl_Raw = repmat(1:Cls,Ns,1);
TrnLbl = Trnbl_Raw(:);
Mdl_HMDA = SetM_HMDA(permute(Trn_Ftr,[1 2 5 3 4]),[7 26 7 size(TrnLbl,1)],'FixTime');
%% 
ACC = Offline_CrossValid(TrainData,Trn_Ftr,RdTrca_idx1);
toc


