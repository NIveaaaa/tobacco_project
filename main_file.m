clear;


diary off
delete myrun.out
diary myrun.out

%% load the data
load('DCE1_workingfile.mat')
load('DCE2_workingfile.mat')

%% global variable

global nresp nalt1 nset1 nalt2 nset2
global d3 d4 d9 lower_index
global SEED NDRAW
global X1reshape X2reshape Y1reshape Y2reshape
global rank1reshape rank2reshape emotionreshape

nalt1 = 3;
nset1 = 5;
nresp =755;
nalt2 = 4;
nset2 = 7;

index1 = [14:21,23]; % att1image_2 ~ att1image_9,att2_text2
index_choice1 = [10,11];

index2 = [18:25,27]; % x attributes for DCE2
index_choice2 = [12,13,14,15];
index_att2 = 11;

% some help vector
d3= eye(3,'logical');
d4 = eye(4,'logical');
d9 = eye(9,'logical');

lower_tindex = tril(reshape(1:81,9,9),-1);
lower_index = 1:81;
lower_index = lower_index(lower_tindex~=0);
SEED = 123;
NDRAW = 100;
%% reshape parameters for DCE1

X1 = DCE1.data(:,index1);
% X1reshape(:,:,i,j) gives 9 by 3 matrix 
X1reshape = reshape(X1',[length(index1),nalt1,nset1,nresp]);

Y1 = DCE1.data(:,index_choice1);
% Y1reshape(:,:,i,j) gives 2 by 3 matrix
Y1reshape = reshape(Y1',[length(index_choice1),nalt1,nset1,nresp]);

rankDCE1 = ones(size(Y1,1),1,"int8")*2;
rankDCE1(Y1(:,1)==1,1)=1;
rankDCE1(Y1(:,2)==1,1)=3;
rank1reshape = reshape(rankDCE1,[nalt1,nset1,nresp]);
% rank 1 : most likely, ranke 2: middle rank 3: least likely, 1 by 3 matrix

%% reshape parameters for DCE2
X2 = DCE2.data(:,index2);
X2reshape=reshape(X2',[length(index2),nalt2,nset2,nresp]);

Y2 = DCE2.data(:,index_choice2);
Y2reshape = reshape(transpose(Y2),[4,nalt2,nset2,nresp]);

emotion = DCE2.data(:,index_att2);
emotionreshape = reshape(emotion,[nalt2,nset2,nresp]);

% rank 1-4: first choice -> last choice
rankDCE2 = ones(size(Y2,1),1,"int8");
rankDCE2(Y2(:,2)==1,1)=2;
rankDCE2(Y2(:,3)==1,1)=3;
rankDCE2(Y2(:,4)==1,1)=4;
rank2reshape = reshape(rankDCE2,[nalt2,nset2,nresp]);

%% unclean unwanted variables
clear X1 X2 Y1 Y2 rankDCE1 rankDCE2 index1 index2 index_att2 index_choice1...
    index_choice2 low_tindex;

%% declare parameters
% model Xint*gamma1 + \sum_k beta_k (Xint*gamma2) + \sum_k beta_k *ep2 +
% ep1, where ep1 ~ N(0,omega1), ep2. ~ N(0, omega2)
% omega1 ~ [1,1-rho,1-rho; 1-rho, 1, 1-rho; 1-rho,1-rho,1]
% omega2 has diagnol elements 1 with covariance rho_{k,l}
% params[1:9] gamma1
% params[10] rho
% params[11:91] gamma2
% params[92:127] rho_{k,l}
% params[128:136] beta

rng(3)
gamma1 = rand(9,1)*0.1;

gamma2 =  rand(9,9)*0.1;
beta = rand(9,1)*0.1;

% construct a positive definite matrix
% rhokl is not correlation between emotion k and l
% need to reconvert 
rhokl = rand(36,1);
L = eye(9);
L(lower_index) = rhokl;
[omega2,~] = corrcov(L*L');

rho_mat = beta*beta'*omega2;

rho = 1 - sum(rho_mat(d9));

param=[gamma1; rho; reshape(gamma2,81,1); rhokl; beta];

clear temp_l rho_mat

%% 


A = [];
b = [];
Aeq = [];
beq = [];

lb = ones(136,1).*(-inf);
lb(10) = 0;
lb(92:127) = ones(36,1).*(-1);
ub = ones(136,1).*inf;
ub(10) = 2;
ub(92:127) = ones(36,1).*(1);


nonlcon = @varcon;
options = optimoptions('fmincon','Display','iter');
[x,fval] = fmincon(@loglik,param,A,b,Aeq,beq,lb,ub,nonlcon,options)
