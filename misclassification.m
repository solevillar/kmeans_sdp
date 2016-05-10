% Author:       Mixon, Villar, Ward.
% Filename:     misclassification.m
% Last edited:  9 May 2016 
% Description:  This function computes the misclassification rate of a
%               clustering with respect to a planted true clustering.
%               Requires CVX in order to solve a k x k linear program that
%               decides the best match between tue labels and assigned 
%               labels.
%               
%             
% Inputs:
%               -labels:
%               
%               A N x 1 array corresponding to true labels of the points
%               in planted clusters.
%
%               -assignment:
%               
%               A N x 1 array corresponding to an assignment of the points
%               in clusters.
%
% Outputs:
%               -misc:
%               The misclassification rate. 
% 
% References:
%
% -------------------------------------------------------------------------
function misc= misclassification(labels, assignment)

k=max(labels);

planted_count=zeros(k,1);
for i=1:k
    planted_count(i)=sum(labels==i);
end

[assign,order]=sort(assignment);
labels=labels(order);

new_count=zeros(k,1);
for i=1:k
    new_count(i)=sum(assign==i);
end

c=zeros(k,k);
idx=0;
for t=1:k
    aux=labels(idx+1:idx+new_count(t));
    idx=idx+new_count(t);
    for i=1:k
        c(t,i)=sum(aux==i)/new_count(t);
    end
end

cvx_begin quiet
variable X(k,k)
maximize(trace(c*X))
subject to 
X*ones(k,1)==ones(k,1);
ones(1,k)*X==ones(1,k);
X>=0;
cvx_end

misc=1-trace(c*X)/k;
end