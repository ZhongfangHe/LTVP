% Estimate a RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt = b_{j,t-1} + N(0,d_jt), j = 1, ..., K
% d_jt = f(z_jt),
% z_jt = (1-rho_j) * u_j + rho_j * z_{j,t-1} + N(0,sj),
%
% use GCK algorithm for zt (adpative MH， multivariate)
% draw static parameters by integrating out bt (adaptive MH, multivariate)
% use ASIS for extra boosting
%
% truncation link: d_jt = w_j * I{z_jt > c_j}


function [pz, ind2, logrw, count, pz_mean, pz_cov] = pz_simulator_LinkConst(y, x, vary,...
    u, rho, v2, pz_old, ind2_old, b0_mean, b0_cov, pstar, AMH_c, ...
    logrw_old, logrw_start, drawi_start, drawi, burnin, pz_mean_old, pz_cov_old)
% Inputs:
%   y: a n-by-1 vector of target,
%   x: a n-by-k matrix of regressors,
%   vary: a n-by-1 vector of target innovation variance,
%   u: a k-by-1 vector of threshold,
%   rho: a k-by-1 vector of AR coef for zt,
%   a2: a k-by-1 vector of scaling factor for zt,
%   v2: a k-by-1 vector of the invariant part of state variances,
%   pz_old: a n-by-k matrix stacking zt from previous sweep,
%   ind2_old: a n-by-k matrix stacking ind2_t from previous sweep,
%   b0_mean: a k-by-1 vector of prior mean of b0 (b0 if conditioned on b0,
%   b0_cov: a k-by-k matrix of prior covariance matrix of b0 (zero if conditioned on b0),
%   pstar: a scalar of optimal MH acceptance rate (0.25),
%   AMH_c: a scalar of the slope for MH update,
%   logrw_old: a n-by-1 vector of RW std from previous sweep,
%   logrw_start: a n-by-1 vector of the immediate start/re-start value of logrw,
%   drawi_start: a n-by-1 vector of the immediate start/re-start time of logrw,
%   drawi: a scaler of the # of the current sweep,
%   burnin: a scalar of the # of burn-ins (to count the MH acceptance),
%   pz_mean_old: a n-by-K matrix stacking the sample mean of zt based on draws up to drawi-1,
%   pz_cov_old: a n-by-1 cell of the sample cov matrix of zt based on draws up to drawi-1,
% Outputs:
%   pz: a n-by-k matrix stacking zt from current sweep,
%   ind2: a n-by-k matrix stacking ind2_t from current sweep,
%   logrw: a n-by-1 vector of log RW std from current sweep,
%   count: a n-by-1 vector of MH acceptance indicator.
%   pz_mean: a n-by-K matrix stacking the sample mean of zt based on draws up to drawi,
%   pz_cov: a n-by-1 cell of the sample cov matrix of zt based on draws up to drawi (for adaptive MH)

[n,K] = size(x);
zmax = 1e50; %bound zt by +/-zmax to avoid numerical issue
maxNum = log(1e10);


%% Setup for adaptive MH 
% pstar = 0.35;
% AMH_c = 1/(K * pstar * (1 - pstar)) + (1-1/K)*sqrt(pi)*exp(0.5*AMH_a*AMH_a)/(sqrt(2)*AMH_a);
% AMH_slope = AMH_c / drawi;
% AMH_intercept = -AMH_slope * pstar;



%% Backward recursion for omega, mu
w_old = ind2_old;
[omega, mu] = MI_backward_recursion(y, x, vary, w_old);


%% Forward simulation of pz by adaptive MH
ind2 = ind2_old;
pz = pz_old;
logrw = logrw_old;
count = zeros(n,1);
pz_mean = pz_mean_old;
pz_cov = pz_cov_old;
for t = 1:n
    % Collect items for t
    xt = x(t,:)';
    yt = y(t);
    sig2t = vary(t);    
    mut = mu(t,:)';
    omegat = omega{t};
    if t == 1
        mtm1 = b0_mean;
        Mtm1 = b0_cov;
    else
        mtm1 = mt;
        Mtm1 = Mt;
    end
    
    
    % Propose zt from a RW
    pzt_old = pz(t,:)';
    if drawi < 100
        A = eye(K);
    else  
        A = pz_cov_old{t} + 1e-6 * eye(K) / drawi;
    end
    if rcond(A) < 1e-15
        A_half = robust_chol(A);
        eta = A_half * randn(K,1);
    else
        eta = mvnrnd(zeros(K,1),A)';
    end 
    pzt_new = pzt_old + exp(logrw(t)) * eta;
    idx_outlier = find(abs(pzt_new) > zmax);
    if ~isempty(idx_outlier)
        nidx = length(idx_outlier);
        for jj = 1:nidx
            pztjj = pzt_new(idx_outlier(jj));
            pztjj_sign = sign(pztjj);
            pzt_new(idx_outlier(jj)) = zmax * pztjj_sign;
        end
    end
    
    

    % Compute log prior prob
    if t == 1
        pztp1 = pz(t+1,:)'; 
        logprior_old = -0.5*sum(pzt_old.^2) + sum(rho.*pzt_old.*pztp1);
        logprior_new = -0.5*sum(pzt_new.^2) + sum(rho.*pzt_new.*pztp1);               
    elseif t == n
        pztm1 = pz(t-1,:)';
        logprior_old = -0.5*sum(pzt_old.^2) + sum(rho.*pzt_old.*pztm1);
        logprior_new = -0.5*sum(pzt_new.^2) + sum(rho.*pzt_new.*pztm1);
    else
        pztm1 = pz(t-1,:)';
        pztp1 = pz(t+1,:)';
        logprior_old = -0.5*sum((pzt_old.^2).*(1+rho.^2)) + sum(rho.*pzt_old.*(pztm1 + pztp1));
        logprior_new = -0.5*sum((pzt_new.^2).*(1+rho.^2)) + sum(rho.*pzt_new.*(pztm1 + pztp1));
    end
    

    % Compute log likelihood
    wt_old = v2 .* double(pzt_old-u>0);
    loglike_old = loglike_MI(yt, xt, mut, omegat, mtm1, Mtm1, sig2t, wt_old); 

    wt_new = v2 .* double(pzt_new-u>0);   
    loglike_new = loglike_MI(yt, xt, mut, omegat, mtm1, Mtm1, sig2t, wt_new); 

    
    % Compute log proposal
    % Not needed for RW proposal
    
    
    % Compute log acceptance prob
    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        pz(t,:) = pzt_new';
        if drawi > burnin
            count(t) = 1;
        end
    end
    ind2(t,:) = v2' .* double(pz(t,:)-u'>0);
    
    
    % Update proposal stdev
    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start(t)/K);
    d = max(ei - ei_start, 20);
    logrwt = logrw_old(t) + AMH_c * (p - pstar)/d;
    if abs(logrwt - logrw_start(t)) > 1.0986 %log(3) ~= 1.0986 
        drawi_start(t) = drawi;
        logrw_start(t) = logrw_old(t);
    end %restart when useful to allow for larger movement    
    logrw(t) = logrwt;
    
    
    % Use the new zt to update the sample covariance
    zt_mean_old = pz_mean_old(t,:)';
    zt_cov_old = pz_cov_old{t};
    zt_mean = (zt_mean_old * (drawi-1) + pz(t,:)') / drawi;
    zt_cov = (drawi - 1) * (zt_cov_old + zt_mean_old * zt_mean_old') / drawi + ...
        pz(t,:)' * pz(t,:) / drawi - zt_mean * zt_mean';
    pz_mean(t,:) = zt_mean';
    pz_cov{t} = zt_cov;
    
    
    % Use the new ind_t to update mt and Mt for use at t+1
    if t < n
        wt = ind2(t,:)';
        [mt, Mt] = Kalman_iteration(mtm1, Mtm1, yt, xt, sig2t, wt);
    end
end



