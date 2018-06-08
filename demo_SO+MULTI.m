clear, clc, 
close all
addpath(genpath('Funcs'));
addpath(genpath('Add'));

%% 1. Parameter Settings
doFrameRemoving = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration
doMAEEval = true;       %Evaluate MAE measure after saliency map calculation
doPRCEval = true;       %Evaluate PR Curves after saliency map calculation
 theta =10;

SRC = 'Data\ECSSD-0502\';       %Path of input images
BDCON = 'Data\BDCON';   %Path for saving bdCon feature image
RES = 'Data\OURS-ECSSD\';       %Path for saving saliency maps
srcSuffix = '.jpg';     %suffix for your input image
if ~exist(BDCON, 'dir')
    mkdir(BDCON);
end
if ~exist(RES, 'dir')
    mkdir(RES);
end

%% 2. Saliency Map Calculation
files = dir(fullfile(SRC, strcat('*', srcSuffix)));
for k=1:length(files)
    disp(k);
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));
    %% Pre-Processing: Remove Image Frames
    srcImg = imread(fullfile(SRC, srcName));
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    %% Segment input rgb image into patches (SP/Grid)
    pixNumInSP =200;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image  
    if useSP
        [superpixels, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [superpixels, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
    meanPos = GetNormedMeanPos(pixelList, h, w);
    bdIds = GetBndPatchIds(superpixels);
    colDistM = GetDistanceMatrix(meanLabCol);
    posDistM = GetDistanceMatrix(meanPos);
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);
  
    %% Clustering
    adjloop=AdjcProloop(superpixels,spNum);
    edges=getedge(spNum,adjloop,meanLabCol);
   
    %% compute affinity matrix
    weights=makeweights(edges,meanLabCol,theta);
    W=adjacency(edges,weights,spNum);
    
    %% 
    lambda=0.8;
    max_L=floor(spNum/2);
    L1=80;L2=50;L3=30;
    [~,evec,~,DD2_i]=make_spectral_analysis(W,max_L,lambda);
    labels_L1=ncut_B(evec(:,1:L1),DD2_i,L1,spNum);
   
    new_labels_L1=labels_L1+spNum;
    [W1,superpixels_L1,pixelList1]=multiScaleW1(spNum,L1,new_labels_L1,noFrameImg,meanLabCol,pixelList,theta);
    meanRgbCol_L1 = GetMeanColor(noFrameImg, pixelList1);
    meanLabCol_L1 = colorspace('Lab<-', double(meanRgbCol)/255);
     
    labels_L2=ncut_B(evec(:,1:L2),DD2_i,L2,spNum);
    new_labels_L2=labels_L2+spNum+L1;
    [W2,superpixels_L2,pixelList2]=multiScaleW2(spNum,L2,new_labels_L2,noFrameImg,meanLabCol,pixelList,theta);
    meanRgbCol_L2 = GetMeanColor(noFrameImg, pixelList2);
    meanLabCol_L2 = colorspace('Lab<-', double(meanRgbCol)/255);
    
    labels_L3=ncut_B(evec(:,1:L3),DD2_i,L3,spNum);
    new_labels_L3=labels_L3+spNum+L1+L2;
    [W3,superpixels_L3,pixelList3]=multiScaleW3(spNum,L3,new_labels_L3,noFrameImg,meanLabCol,pixelList,theta);
    meanRgbCol_L3 = GetMeanColor(noFrameImg, pixelList3);
    meanLabCol_L3 = colorspace('Lab<-', double(meanRgbCol)/255);
    
    [bgProb, ~, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
    
    %%
    sonum=spNum+L1+L2+L3;  %求超像素层与3个区域层之间的权重
    Wh=zeros(sonum,sonum);
    Wh(1:spNum,1:spNum)=W;
    Wh(spNum+1:spNum+L1,spNum+1:spNum+L1)=W1;
    Wh(spNum+L1+1:spNum+L1+L2,spNum+L1+1:spNum+L1+L2)=W2;
    Wh(spNum+L1+L2+1:spNum+L1+L2+L3,spNum+L1+L2+1:spNum+L1+L2+L3)=W3;
    
    bgWeightS=zeros(sonum,1);
    bgWeightS(1:spNum,1)=bgWeight;
  
    fgWeight=zeros(sonum,1);
    fgWeight(1:spNum,1)=wCtr;
    
    %% 超像素层与区域层之间的权重 %% 
   
% %     gama1=16.5;
% %     gama2=16.5;
% %     gama3=16.5;
%     for index=1:spNum       
% % %         valDistances1=sqrt(sum((meanRgbCol_L1(labels_L1(index),:)-meanRgbCol(index,:)).^2,2));
% % %         gama1=exp(-theta*valDistances1);
%         gama1=18.5*exp(sum(-(sqrt((meanLabCol_L1(labels_L1(index),:)-meanLabCol(index,:)).^2))/theta.^2));
%         Wh(index,new_labels_L1(index))=gama1;
%         Wh(new_labels_L1(index),index)=gama1;
% %         
% % %         valDistances2=sqrt(sum((meanRgbCol_L2(labels_L2(index),:)-meanRgbCol(index,:)).^2,2));
% % %         gama2=exp(-theta*valDistances2);
%         gama2=17.5*exp(sum(-(sqrt((meanLabCol_L2(labels_L2(index),:)-meanLabCol(index,:)).^2))/theta.^2));
%         Wh(index,new_labels_L2(index))=gama2;
%         Wh(new_labels_L2(index),index)=gama2;
%         
% % %         valDistances3=sqrt(sum((meanRgbCol_L2(labels_L2(index),:)-meanRgbCol(index,:)).^2,2));
% % %         gama3=exp(-theta*valDistances3);
%         gama3=16.5*exp(sum(-(sqrt((meanLabCol_L3(labels_L3(index),:)-meanLabCol(index,:)).^2))/theta.^2));
%         Wh(index,new_labels_L3(index))=gama3;
%         Wh(new_labels_L3(index),index)=gama3;
%      end
    
    gama=8.5;
    for index=1:spNum  
        Wh(index,new_labels_L1(index))=gama;
        Wh(new_labels_L1(index),index)=gama;
        Wh(index,new_labels_L2(index))=gama;
        Wh(new_labels_L2(index),index)=gama;
        Wh(index,new_labels_L3(index))=gama;
        Wh(new_labels_L3(index),index)=gama;
    end
    
    %% Saliency Optimization 
    optwCtr = SaliencyOptimization(Wh, bgWeightS, fgWeight);    
    saliency=optwCtr(1:spNum);
%     min_saliency= min(saliency);
%     real_min_saliency=min(min_saliency);
%     max_saliency = max(saliency);
%     real_max_saliency=min(max_saliency);
%     real_saliency = (saliency- real_min_saliency) / (real_max_saliency- real_min_saliency + eps);
%     
%     saliency1=optwCtr(spNum+1:spNum+L1);
%     min_saliency1= min(saliency1);
%     real_min_saliency1=min(min_saliency1);
%     max_saliency1 = max(saliency1);
%     real_max_saliency1=min(max_saliency1);
%     real_saliency1 = (saliency1- real_min_saliency1) / (real_max_saliency1- real_min_saliency1 + eps);
%     
%     saliency2=optwCtr(spNum+L1+1:spNum+L1+L2);
%     min_saliency2= min(saliency2);
%     real_min_saliency2=min(min_saliency2);
%     max_saliency2 = max(saliency2);
%     real_max_saliency2=min(max_saliency2);
%     real_saliency2 = (saliency2- real_min_saliency2) / (real_max_saliency2- real_min_saliency2 + eps);
%     
%     saliency3=optwCtr(spNum+L2+L1+1:spNum+L1+L2+L3);
%     min_saliency3= min(saliency3);
%     real_min_saliency3=min(min_saliency3);
%     max_saliency3 = max(saliency3);
%     real_max_saliency3=min(max_saliency3);
%     real_saliency3 = (saliency3- real_min_saliency3) / (real_max_saliency3- real_min_saliency3 + eps);
% 
% 
%    % [hh,ww]=size(adjcMatrix);
%     imsal1=zeros(h,w);
%     for i=1:spNum
%         imsal1(pixelList{i})=real_saliency(i);
%     end
%     
%     imsal2=zeros(h,w);
%     for i=1:L1
%         imsal2(pixelList1{i})=real_saliency1(i);
%     end
%     
%     imsal3=zeros(h,w);
%     for i=1:L2
%         imsal3(pixelList2{i})=real_saliency2(i);
%     end
%     
%     imsal4=zeros(h,w);
%     for i=1:L3
%         imsal4(pixelList3{i})=real_saliency3(i);
%     end
%     
%     imsal=imsal1*1+imsal2*0.5+imsal3*0.4+imsal4*0.3;


    smapName=fullfile(RES, strcat(noSuffixName, '_sal.png'));
%          imwrite(imsal,smapName,'png');
    SaveSaliencyMap(saliency, pixelList, frameRecord, smapName, true)

end
% %画柱状图
% % function Plot_PreRecallThousand_contrast
% %
% clear;
% A=xlsread('P_R_F and MAE_saliency.xls');
% a=A(26,1:3);
% %存放方法图像
% Mapdir ='C:\Users\QIAN CHEN\Documents\MATLAB\Anastasia\CVPR14_saliencyoptimization\Data\OursMSRA';
% %存放真值图像
% GroundDir = 'C:\Users\QIAN CHEN\Documents\MATLAB\Anastasia\CVPR14_saliencyoptimization\Data\MSRA1K-正确\MSRA 1K_GT';
% cd (GroundDir);
% ImgEnum=dir('*.png');   ImgNum=length(ImgEnum);%图像数量
% Pre = zeros(21,1);
% Recall = zeros(21,1);
% FMeasure = zeros(21,1);
% jj=0;    PreF = 0; RecallF = 0; FMeasureF = 0;   FigAnd = 0;  bigthreshold = 0; ThresholdAnd = 0;
% for i=1:ImgNum
%     cd(GroundDir);
%     Binary = imread( ImgEnum(i).name );
%     NumOne= length( find(Binary(:,:,1) >0) );   
%     [height,width] = size( Binary(:,:,1) );
%     cd (Mapdir);
%     mapImg = imread( strcat( ImgEnum(i).name(1:end-4),'_sal.png' ) );
%     %     Label2 = imread( ImgEnum(i).name );
%     Label1 = mapImg;
%     Label2 = mat2gray(Label1  );   
%     %%  thou berke Pre recall
%     if NumOne ~= 0
%         jj=jj+1;   mm = 1;
%         for j = 0 : .05 : 1
%             Label3 = zeros( height, width );
%             Label3( Label2>=j )=1;
%             NumRec = length( find( Label3==1 ) );            
%             LabelAnd = Label3 & Binary(:,:,1);
%             NumAnd = length( find( LabelAnd==1 ) );
%             if NumAnd == 0
%                 FigAnd = FigAnd + 1;
%                 break;
%             end
%             Pretem = NumAnd/NumRec;
%             Recalltem =  NumAnd/NumOne;           
%             Pre(mm) = Pre(mm) +  Pretem;
%             Recall(mm) = Recall(mm) + Recalltem;            
%             FMeasure(mm) = FMeasure(mm) + ( (1 + .3) * Pretem * Recalltem ) / ( .3 * Pretem + Recalltem );
%             mm = mm + 1;
%         end        
%         sumLabel =  2* sum( sum(Label2) ) / (height*width) ;
%         if ( sumLabel >= 1 )
%             sumLabel = .902 ;    bigthreshold = bigthreshold +1;
%         end       
%         Label3 = zeros( height, width );
%         Label3( Label2>=sumLabel ) = 1;       
%         NumRec = length( find( Label3==1 ) );       
%         LabelAnd = Label3 & Binary(:,:,1);
%         NumAnd = length( find ( LabelAnd==1 ) );       
%         if NumAnd == 0
%             ThresholdAnd = ThresholdAnd +1;
%             continue;
%         end        
%         PreFtem = NumAnd/NumRec;
%         RecallFtem = NumAnd/NumOne;        
%         PreF = PreF +    PreFtem;
%         RecallF = RecallF +    RecallFtem;                                                                                                                                                                                                                        
%         
%         
%         FMeasureF = FMeasureF + ( ( ( 1 + .3) * PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) );
%     end   
% end
% 
% %% Mean Pre Recall
% FigAnd;
% ThresholdAnd
% bigthreshold
%  cd 'C:\Users\QIAN CHEN\Documents\MATLAB\Anastasia\CVPR14_saliencyoptimization\Data\MSRA1K-正确\MSRA 1K_GT';
% Pre = Pre ./jj ;
% Recall = Recall ./jj;
% FMeasure = FMeasure ./ jj;
% 
% PreF = PreF /jj
% RecallF = RecallF /jj
% FMeasureF = FMeasureF / jj
% data1=[PreF,RecallF,FMeasureF];
% real_data=[data1;a];
% figure(1);
% bar(real_data);

% %画曲线图
% GT='F:\MATLAB\Anastasia\CVPR14_saliencyoptimization\Data\GT-ECSSD';
% gtSuffix='.png';
% xSuffix='.png';
% hold on;
% RESpath1='F:\MATLAB\CVPR14_saliencyoptimization原来的版本\Data\2014ECSSD';
% RESpath2='F:\MATLAB\Anastasia\CVPR14_saliencyoptimization\Data\Data\OursEcssd';
% [rec1, prec1]=DrawPRCurve(RESpath1,'_sal.png',GT,gtSuffix,true,true);
% [rec2, prec2]=DrawPRCurve(RESpath2,'_sal.png',GT,gtSuffix,true,true);
% 
% figure(1)
% plot(rec1, prec1, 'b', 'linewidth', 2);
%  hold on;
%  plot(rec2, prec2, 'r', 'linewidth', 2);
% hold on;