function [data,TrainFtr] = data_prepocess(datapath,Fs,time,n_block,cns,re_Fs,bp_fltr,wavename,F_Range,df)
primay_dir = pwd;
cd([datapath]);
block_name = dir('*.cnt');
block_name = {block_name.name};
label_cashe = [];
block_label_cashe=[];
TrainData=[];
for block = 1:n_block
    eeg = pop_loadcnt(block_name{block},'dataformat','int32');
    eeg_raw = eeg.data(cns,:);
    event=eeg.event;
    for trl=1:length(event)
        a(trl)=event(1,trl).type;
        b(trl)=event(1,trl).latency;
    end
    latency=round(b);
    Label=[1 2 3 4 5 6 7 8 9];
    [N,Wn]=cheb1ord([1 bp_fltr]/Fs*2,[0.1 40]/Fs*2,4,10);
    [f_b,f_a]=cheby1(N,0.5,Wn);
    DataPro=[];
    for cls=1:size(Label,2)
        label_loc=find(a==Label(1,cls));
        for num=1:length(label_loc)
            RDataPro_cashe=[];
            DataFtr_cashe=[];
            DataRmb_cashe=[];
            DataRrf_cashe=[];
            RDataPro_cashe=eeg_raw(:,latency(label_loc(num))-Fs*(time(1,1)+time(1,2)):latency(label_loc(num))+Fs*(time(1,3)));% assume the label were printed in the end of trials
            DataFtr_cashe=filtfilt(f_b,f_a,double(RDataPro_cashe'));
            DataRmb_cashe=resample(DataFtr_cashe,1,Fs/re_Fs);
            %% special for cns9 un-reference
            DataPro(:,:,num,cls)=DataRmb_cashe([end-re_Fs*(time(1,2)+time(1,3)):end],1:end)';
        end
    end
    label_cashe=cat(2,label_cashe,a);
    TrainData=cat(3,DataPro,TrainData);
    block_label_cashe=cat(2, block_label_cashe, block * ones(1, length(a)));
end
data{1} = TrainData;
%%
fmax = F_Range(1,2);
fmin = F_Range(1,1);
totalscal = (fmax-fmin)/df;
f = fmin:df:fmax-df;%预期的频率
wcf = centfrq(wavename); %小波的中心频率
scal = re_Fs*wcf./f;
[Nc, Nt, Ns, Cls] = size(TrainData);
%%
for cls = 1:Cls
    for trl = 1:Ns
        for cns = 1:Nc         
            coefs(:,:,cns,trl,cls) = cwt(squeeze(TrainData(cns,:,trl,cls)),scal,wavename);
        end
    end
end
TrainFtr = real(permute(coefs,[3 2 4 5 1]));
cd(primay_dir);
end