% set up constraint to ganrantee var(\sum_k beta_k ep_2 k + ep_1) =1 

function [c,ceq] = varcon(x)
global lower_index d9
rho = x(10);
beta = x(1:9);
rhokl = x(92:127);
L = eye(9);
L(lower_index) = rhokl;
[omega2,~] = corrcov(L*L');

rho_mat = beta*beta'*omega2;
c = [];
ceq = 1-sum(rho_mat(d9))-rho*rho;
