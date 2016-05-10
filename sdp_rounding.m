% Author:       Mixon, Villar, Ward.
% Filename:     sdp_rounding.m
% Last edited:  9 May 2016 
% Description:  This function applies the rounding procedure from [1] to 
%               denoised points coming from the application of kmeans SDP.
%                            
% Inputs:
%               -denoised:
%               
%               A d x N array corresponding to coordinates of denoised 
%               points
%
%               -k:
%               
%               Number of clusters.
%
% Outputs:
%               -centers:
%               
%               The k most frequent denoised points. 
%
%               -assignment:
% 
%               The assignment of points to centers (clusters).
%
% References:
%
% [1] Mixon, Villar, Ward. Clustering subgaussian mixtures via semidefinite
%       programming
%
% -------------------------------------------------------------------------

function [centers, assignment]=sdp_rounding(denoised, k)
N=size(denoised,2);

% computation of an affinity matrix identifying repeated denoised points
affinity=zeros(N,N);
for i=1:N
    for j=i:N
        if norm(denoised(:,i)-denoised(:,j))<1e-3
            affinity(i,j)=1;
            affinity(j,i)=1;
        end
    end
end

% centers are k most popular points
centers=zeros(k,k);
for t=1:k
    [~, idx]=max(sum(affinity));
    aux=affinity(:,idx);
    centers(t,:)=denoised(:, idx)';
    for i=1:N
        if aux(i)==1
            affinity(i,:)=zeros(1,N);
            affinity(:,i)=zeros(N,1);
        end
    end
end

% assignment of points to closest center
ind=zeros(N,1);
for i=1:N
    aux=zeros(k,1);
    for t=1:k
        aux(t,1)=norm(denoised(:,i)'- centers(t,:));
    end
    [~, ind(i,1)]= min(aux);
end
assignment=ind;
end