%Calculate loglikelihood function for probit model
%Inputs: coef is Kx1 where K is number of explanatory variables
%
%Outputs: negative loglikelihood
%Globals: nresp: scalar number, number of subjects
%         nalt1: scalar number of alternatives per set in DCE1
%         nset1: scalar number of sets in DCE1
%         nalt2: ... for DCE2
%         nset2: ... for DCE2
%         d3, d4, d9, lower_index: some helper value
%         NDRAW SEED: number of draws and seed
%         X1reshape: X1 matrix (reshaped)
%         X2reshape: X2 matrix (reshaped)
%         emotionreshape: emation of DCE2
%         rank1reshape: ranked choice of DCE1
%         rank2reshape: ranked choice of DCE2


function ll=loglik(param);

global nresp nalt1 nset1 nalt2 nset2 d3 d4 d9 lower_index SEED NDRAW
global X1reshape X2reshape emotionreshape rank1reshape rank2reshape

%Arrange parameters
% params[1:9] gamma1
% params[10] rho
% params[11:91] gamma2
% params[92:127] rho_{k,l}
% params[128:136] beta
gamma1 = param(1:9);
rho = param(10);
gamma2 = reshape(param(11:91),9,9);
rhokl = param(92:127);

L = eye(9);
L(lower_index) = rhokl;
[omega2,~] = corrcov(L*L');


beta = param(128:136);
p1 = zeros(nresp,nset1);
p2 = zeros(nresp,nset2);

%rng(SEED);
for id = 1:nresp
    for s = 1:nset2
        
        M2 = generateM(rank2reshape(:,s,id));

        % generate L for ghk simulator
        emotion_obs = emotionreshape(:,s,id);
        omega_obs = omega2(emotion_obs,emotion_obs);
        omega_alt = M2*omega_obs*M2';
        L2 = chol(omega_alt,'lower');
        
        % generate Vtilde for ghk simulator
        Xint2 = X2reshape(:,:,s,id)';
        Vint2 = Xint2*gamma2(:,emotion_obs);
        Vint2 = Vint2(d4); % final ultily 
        V2tilde = M2*Vint2;
        
        e = randn(nalt2-1,NDRAW);
        p2(id,s) = ghk(V2tilde,L2,e);
    end
    
    for s = 1:nset1
        M1 = generateM(rank1reshape(:,s,id));
        Xint1 = X1reshape(:,:,s,id)';
        V1tilde = M1 * (Xint1*gamma1 + sum(Xint1*beta,2));


        %rho_mat = beta*beta'*omega2;
        %rho = 1 - sum(rho_mat(d9));
        omega1 = ones(nalt1).*(1-rho);
        omega1(d3) = 1;
        omega1_alt = M1*omega1*M1';
        L1 = chol(omega1_alt,'lower');
        
        e = randn(nalt1-1,NDRAW);
        p1(id,s) = ghk(V1tilde,L1,e);

    end
    
        
end

p1 = max(p1,0.00000001);
p2 = max(p2,0.00000001); %As a precaution
ll=-sum(log(p2),[1,2])-sum(log(p1),[1,2]);  %Negative since neg of ll is minimized


