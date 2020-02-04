% generating transition matrix M
% Input: 

function M=generateM(rankedres)
l = length(rankedres);
[~,rank] = sort(rankedres);
alt2 = 1:l;
ranked = alt2(rank);

M = zeros(l-1,l);

for i = 1:(l-1)
    M(i,ranked(i))=-1;
    M(i,ranked(i+1))=1;
end
