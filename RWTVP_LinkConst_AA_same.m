% Estimate a RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt = b_{j,t-1} + N(0,d_jt), j = 1, ..., K
% d_jt = vj2 * I(z_jt > c_j),
% z_jt = (1-rho_j) * u_j + rho_j * z_{j,t-1} + N(0,sj),
%
% use GCK algorithm for zt (adpative MH， multivariate)
% draw static parameters by integrating out bt (adaptive MH, multivariate)
% use ASIS for extra boosting




function draws = RWTVP_LinkConst_AA_same(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws after burnin
%   ind_SV: an indicator if SV for measurement noise variance
%   ind_sparse: an indicator if sparsifying is performed (not applicable here; always set 0)
%   ind_forecast: an indicator if Kalman filter is run for subsequent forecasts
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
% minNum = 1e-100;
% maxNum = log(1e10);


%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
phil_d = 1./gamrnd(0.5,1,K,1);
phil = 1./gamrnd(0.5*ones(K,1),phil_d); %local variances
taul_d = 1/gamrnd(0.5,1);
taul = 1/gamrnd(0.5, taul_d); %global variance
psil = taul*phil; 
beta0 = sqrt(psil) .* randn(K,1);


%% Priors: scaling factor for state noise, v ~ N(0, diag(psi)) 
psi = 0.01*ones(K,1); %default
v = sqrt(psi).*randn(K,1);



%% Priors: stationary prob of zero for zt
q1 = 1; q2 = 1; %uniform
q = betarnd(q1,q2,K,1);
qq = log(q) - log(1-q); 



%% Priors: slope of zt
r1 = 20; r2 = 1.5; %narrow
rho = 2*betarnd(r1,r2)-1;
r = log(1+rho)-log(1-rho);



%% Initialize latent index
pz = zeros(n,K);
pz(1,:) = (1/sqrt(1-rho^2))*randn(1,K);
for t = 2:n
    pz(t,:) = rho*pz(t-1,:) + randn(1,K);
end
u = norminv(q)/sqrt(1-rho^2); %threshold of zt
ind2 = double(pz-repmat(u',n,1)>0);


%% Initiaze state variance
state_var = cell(n,1);
for t = 1:n
    state_var{t} = diag(ind2(t,:));
end %covar matrices of state noise for simulation smoother (AA)


%% Set up adaptive MH
pstar = 0.25; %multivariate MH
tmp_const = -norminv(0.5*pstar);
AMH_c = 1/(K * pstar * (1-pstar)) + (1-1/K)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;

pstar_r = 0.44; %univariate MH
AMH_cr = 1/(pstar_r * (1-pstar_r)); 

logrw = zeros(n,1);
logrw_start = logrw;
drawi_start = zeros(n,1); %z

% logrw_v = 0;
% logrw_start_v = logrw_v;
% drawi_start_v = 0; %v

% logrw_beta0 = 0;
% logrw_start_beta0 = logrw_beta0;
% drawi_start_beta0 = 0; %beta0

logrw_q = 0;
logrw_start_q = logrw_q;
drawi_start_q = 0; %q

logrw_r = -1;
logrw_start_r = logrw_r;
% drawi_start_r = 0; %rho


%% Priors: SV or constant measurement noise variance
if ind_SV == 1
    % long-run mean: p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
    % persistence: p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
    % variance: p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)
    muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
    phih0 = 0.95; invVphih = 1/0.04; % AR(1): p(phi) ~ N(phi0, Vphi)I(-1,1)
    priorSV = [muh0 invVmuh phih0 invVphih 0 0]'; %collect prior hyperparameters
    muh = muh0 + sqrt(1/invVmuh) * randn;
    phih = phih0 + sqrt(1/invVphih) * trandn((-1-phih0)*sqrt(invVphih),(1-phih0)*sqrt(invVphih));

    lambdah = 1/gamrnd(0.5,1);
    sigh2_s = 1/gamrnd(0.5,lambdah);
    sigh2 = gamrnd(0.5,2*sigh2_s);
    sigh = sqrt(sigh2);

    hSV = log(var(y))*ones(n,1); %initialize by log OLS residual variance.
    vary = exp(hSV);
else %Jeffery's prior p(sig2) \prop 1/sig2
    sig2 = var(y); %initialize
    vary = sig2 * ones(n,1);
end 


%% MCMC
draws.taul = zeros(ndraws,2);
draws.phil = zeros(ndraws,2*K);
draws.beta0 = zeros(ndraws,K); 
% draws.count_beta0 = 0;
% draws.logrw_beta0 = zeros(ndraws,1); %beta0

draws.v = zeros(ndraws,K);
% draws.count_v = 0;
% draws.logrw_v = zeros(ndraws,1); %v

draws.u = zeros(ndraws,K);
draws.q = zeros(ndraws,K);
draws.count_q = 0;
draws.logrw_q = zeros(ndraws,1); %q

draws.rho = zeros(ndraws,1);
draws.count_r = 0; 
draws.logrw_r = zeros(ndraws,1);%rho

draws.beta = cell(K,1);
for j = 1:K
    draws.beta{j} = zeros(ndraws,n);
end %beta

draws.count_pz = zeros(n,1);
draws.logrw = zeros(ndraws,n);
draws.pz = cell(K,1);
draws.ind2 = cell(K,1);
for j = 1:K
    draws.pz{j} = zeros(ndraws,n);
    draws.ind2{j} = zeros(ndraws,n);
end %z

if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K);
    draws.beta0_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K,1);
    for j = 1:K
        draws.beta_sparse{j} = zeros(ndraws,n);
    end
end

if ind_forecast == 1
    draws.bn_mean = zeros(ndraws,K);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K,K);
    end
    if ind_sparse == 1
        draws.bn_smean = zeros(ndraws,K);
        draws.bn_scov = cell(ndraws,1);
        for j = 1:ndraws
            draws.bn_scov{j} = zeros(K,K);
        end
    end    
end

tic;
ntotal = burnin + ndraws;
pz_mean = zeros(n,K);
pz_cov = cell(n,1);
for t = 1:n
    pz_cov{t} = zeros(K,K);
end
% v_mean = zeros(K,1);
% v_cov = zeros(K,K);
% beta0_mean = zeros(K,1);
% beta0_cov = zeros(K,K);
q_mean = zeros(K,1);
q_cov = zeros(K,K);
% r_mean = zeros(1,1);
% r_cov = zeros(1,1);
beta0_star = zeros(K,1);
for drawi = 1:ntotal   
    % Draw indicator (AA, integrating out beta， multivariate)
    b0_mean = beta0_star;
    b0_cov = zeros(K,K);    
    pz_mean_old = pz_mean;
    pz_cov_old = pz_cov;
    pz_old = pz;
    ind2_old = ind2;
    logrw_old = logrw;  
    yy = y - x * beta0;
    xx = x.*repmat(v',n,1);
    v2tmp = ones(K,1);
    [pz, ind2, logrw, count_pz, pz_mean, pz_cov] = pz_simulator_LinkConst(yy, xx,...
        vary, u, rho*ones(K,1), v2tmp, pz_old, ind2_old, b0_mean, b0_cov, pstar, AMH_c, logrw_old,...
        logrw_start, drawi_start, drawi, burnin, pz_mean_old, pz_cov_old); 

    
    % Draw rho (AA, integrating out beta， multivariate)
    count_r = 0;

    rho_old = rho;
    rho2_old = rho^2;
    r_old = r;
%     if drawi < 100
%         A = eye(1);
%     else  
%         A = r_cov + 1e-6 * eye(1) / drawi; %add a small constant
%     end
%     eps = mvnrnd(zeros(1,1),A)'; %correlated normal
%     eps = sqrt(A)*randn;
    eps = randn;
    r_new = r_old + exp(logrw_r) * eps;
    rho_new = (1-exp(-r_new)) / (1+exp(-r_new));
    rho2_new = rho_new^2;
    
    logprior_old = r1*sum(log(1+rho_old)) + r2*sum(log(1-rho_old));
    logprior_new = r1*sum(log(1+rho_new)) + r2*sum(log(1-rho_new)); %p(r)
    
    ztrans = [sqrt(1-rho2_old)*pz(1,:); pz(2:n,:) - rho_old*pz(1:n-1,:)];
    z2trans = ztrans.^2;
    loglike1_old = 0.5*K*log(1-rho2_old) - 0.5*sum(sum(z2trans));
    ztrans = [sqrt(1-rho2_new)*pz(1,:); pz(2:n,:) - rho_new*pz(1:n-1,:)];
    z2trans = ztrans.^2;
    loglike1_new = 0.5*K*log(1-rho2_new) - 0.5*sum(sum(z2trans)); %p(z|r)    

    w_old = ind2;
    loglike2_old = loglike_TVP2(yy, xx, vary, w_old, beta0_star);
    u_new = norminv(q)/sqrt(1-rho2_new);
    w_new = double(pz-repmat(u_new',n,1)>0);
    loglike2_new = loglike_TVP2(yy, xx, vary, w_new, beta0_star); %p(y|z,r)

    logprob = logprior_new + loglike1_new + loglike2_new ...
        - logprior_old - loglike1_old - loglike2_old;
    if log(rand) <= logprob
        rho = rho_new;
        r = r_new;
        ind2 = w_new;
        loglike_integ = loglike2_new;
        if drawi > burnin
            count_r = 1;
        end
    else
        loglike_integ = loglike2_old;
    end
     

    p = exp(min(0,logprob));
%     ei = max(200, drawi);
%     ei_start = max(200, drawi_start_r);
%     d = max(ei - ei_start, 20);
    d = drawi;
    logrwj = logrw_r + AMH_cr * (p - pstar_r)/d;
    if abs(logrwj - logrw_start_r) > 1.0986 %log(3) ~= 1.0986 
%         drawi_start_r = drawi;
        logrw_start_r = logrw_r;
    end %restart when useful to allow for larger movement    
    logrw_r = logrwj; %update proposal stdev
    
    
%     r_mean_old = r_mean;
%     r_cov_old = r_cov;
%     r_mean = (r_mean_old * (drawi-1) + r) / drawi;
%     r_cov = (drawi - 1) * (r_cov_old + r_mean_old * r_mean_old') / drawi + ...
%         r * r' / drawi - r_mean * r_mean'; %update the sample covariance
    
    
    
    % Draw q (AA, integrating out beta， multivariate)
    count_q = 0;

    q_old = q;
    qq_old = qq;
    if drawi < 100
        A = eye(K);
    else  
        A = q_cov + 1e-6 * eye(K) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(K,1),A)'; %correlated normal
    qq_new = qq_old + exp(logrw_q) * eps; 
    q_new = 1./(1+exp(-qq_new));
    
    logprior_old = q1*sum(log(q_old)) + q2*sum(log(1-q_old));
    logprior_new = q1*sum(log(q_new)) + q2*sum(log(1-q_new)); %p(qq)

%     w_old = ind2;
%     loglike_old = loglike_TVP2(y, x, vary, w_old, beta0);
    loglike_old = loglike_integ;
    u_new = norminv(q_new)/sqrt(1-rho^2);
    w_new = double(pz-repmat(u_new',n,1)>0);
    loglike_new = loglike_TVP2(yy, xx, vary, w_new, beta0_star); %p(y|qq)

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        q = q_new;
        qq = qq_new;
        ind2 = w_new;
%         loglike_integ = loglike_new;
        if drawi > burnin
            count_q = 1;
        end
%     else
%         loglike_integ = loglike_old;
    end
     

    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start_q/K);
    d = max(ei - ei_start, 20);
    logrwj = logrw_q + AMH_c * (p - pstar)/d;
    if abs(logrwj - logrw_start_q) > 1.0986 %log(3) ~= 1.0986 
        drawi_start_q = drawi;
        logrw_start_q = logrw_q;
    end %restart when useful to allow for larger movement    
    logrw_q = logrwj; %update proposal stdev
    
    
    q_mean_old = q_mean;
    q_cov_old = q_cov;
    q_mean = (q_mean_old * (drawi-1) + qq) / drawi;
    q_cov = (drawi - 1) * (q_cov_old + q_mean_old * q_mean_old') / drawi + ...
        qq * qq' / drawi - q_mean * q_mean'; %update the sample covariance
    
    
    % Draw beta_star (AA) 
    for t = 1:n
        state_var{t} = diag(ind2(t,:));
    end %covar matrices of state noise for simulation smoother (AA)    
    beta_star = Simulation_Smoother_DK(yy, xx, vary, state_var(2:n),...
        beta0_star, state_var{1});      
   

    % Draw v, beta0 based on beta_star (AA)
    x_beta0v2 = [x  x.* beta_star];

    psi_beta0v = [psil; psi];
    sigy = sqrt(vary);
    xstar = x_beta0v2 ./ repmat(sigy,1,K+K);
    ystar = y ./ sigy;
    A_inv = diag(1./psi_beta0v) + xstar' * xstar;
    if rcond(A_inv) > 1e-15
        A_inv_half = chol(A_inv);
        a_beta0v = A_inv \ (xstar' * ystar);
        beta0v = a_beta0v + A_inv_half \ randn(K+K,1);
    else
        A_beta0v = robust_inv(A_inv);
        A_half = robust_chol(A_beta0v);
        a_beta0v = A_beta0v * (xstar' * ystar);
        beta0v = a_beta0v + A_half * randn(K+K,1);
    end
    beta0 = beta0v(1:K);
    v = beta0v(K+1:K+K);
%     v2 = v.^2;


    % ASIS: compute beta
    beta = beta_star .* repmat(v',n,1) + repmat(beta0',n,1);
    diff_beta = [beta(1,:)-beta0'; beta(2:n,:) - beta(1:(n-1),:)];
    diff_beta2 = (diff_beta.^2);
    
    
    % ASIS: update v2 (SA)
    v2 = v.^2;
    v_sign = sign(v);
    for j = 1:K
        idx = find(ind2(:,j)==1); %remove constant betas
        nn = length(idx);
        if nn > 0
            dd = sum(diff_beta2(idx,j));
            [v2(j),~] = gigrnd(0.5-0.5*nn, 1/psi(j), dd, 1);
%             if v2(j) == 0
%                 v2(j) = minNum;
%             end
%             if isnan(v2(j))
%                 error('v2(j) is nan');
%             end
        end
    end
    v = sqrt(v2) .* v_sign;
    
    
    % ASIS: update beta0 (SA)
    for j = 1:K
        if ind2(1,j) == 1
            jvar_inv = 1/psil(j) + 1/v2(j);
            jvar = 1/jvar_inv;
            jmean = jvar*beta(1,j)/v2(j);
            beta0(j) = jmean + sqrt(jvar)*randn;
        end
    end   
      
    
    % ASIS: compute back beta_star
    beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);
    

    % Asis-z if q is uniform prior
    if and(q1==1, q2==1)
        % Asis-z: compute zhat
        u = norminv(q)/sqrt(1-rho^2);
        zhat = pz - repmat(u',n,1);

        % Asis-z: redraw q by independent MH
        for j = 1:K
            rhoj = rho;
            xtmp = -[1; ones(n-1,1)*(1-rhoj)/sqrt(1-rhoj^2)];
            ytmp = [sqrt(1-rhoj^2)*zhat(1,j); zhat(2:n,j)-rhoj*zhat(1:n-1,j)];
            uu_var_inv = 1 + xtmp' * xtmp;
            uu_var = 1/uu_var_inv;
            uu_mean = uu_var * xtmp' * ytmp;
            uu = uu_mean + sqrt(uu_var)*randn; 
            q(j) = normcdf(uu);
        end

        % Asis-z: compute back z
        u = norminv(q)/sqrt(1-rho^2);
        pz = zhat + repmat(u',n,1);        
    end    
    
    
    % Finalize the beta of the sweep
    u = norminv(q)/sqrt(1-rho^2);
    ind2 = double(pz-repmat(u',n,1)>0);
    
    % Update hyperparameters of v
%     tmp = v.^2; 
%     [tau, tau_d, phi, phi_d] = Horseshoe_update_vector_scaled(tmp, tau, tau_d, ...
%         phi, phi_d, 1/n); 
%     psi = tau * phi;
%     psi = 1./gamrnd(v01+0.5, 1./(v02+0.5*v2)); %IG prior
    
    
    % Update hyperparameters of beta0
    tmp = beta0.^2; 
    [taul, taul_d, phil, phil_d] = Horseshoe_update_vector_scaled(tmp, taul, taul_d, ...
        phil, phil_d, 1/n);   
    psil = taul * phil;    
    
    
    % Residual variance
    yfit = sum(x .* beta,2);
    eps = y - yfit;
    if ind_SV == 1
        logz2 = log(eps.^2 + 1e-100);
        [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
            muh, phih, sigh, sigh2_s, lambdah, priorSV);    
        vary = exp(hSV); 
    else
        sig2 = 1/gamrnd(0.5*n, 2/(eps'*eps));
        vary = sig2 * ones(n,1); 
    end  
    
    
    % Sparsify beta if needed
    if ind_sparse == 1
        z = x .* beta_star;
        v_sparse = SAVS_vector(v, z); 
        
        beta0_sparse = SAVS_vector(beta0,x);
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);          
    end
    
    
    % Compute mean and covar of p(bn|y1,...,yn) for subsequent forecasts
    if ind_forecast == 1 
        v2 = v.^2;
        for t = 1:n
            state_var{t} = diag(v2'.*ind2(t,:));
        end %covar matrices of state noise for simulation smoother (SA)        
        P1 = state_var{1};
        a1 = beta0; %bstar1 = 0
        [bn_mean, bn_cov] = Kalman_filter_robust(y, x, ...
            vary, state_var(2:n), a1, P1);                  
        if ind_sparse == 1
            bn_smean = v_sparse.*bstarn_mean+ beta0_sparse;
            bn_scov = (v_sparse*v_sparse') .* bstarn_cov;
        end       
    end    
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K
            draws.beta{j}(i,:) = beta(:,j)';
        end   
        
        draws.q(i,:) = q';
        draws.u(i,:) = u';
        draws.logrw_q(i) = logrw_q;
        draws.count_q = draws.count_q + count_q; %q
        
        draws.rho(i,:) = rho';
        draws.logrw_r(i) = logrw_r;
        draws.count_r = draws.count_r + count_r;%rho
        
        draws.v(i,:) = v';  
%         draws.count_v = draws.count_v + count_v; 
%         draws.logrw_v(i) = logrw_v;%v        
        
        draws.beta0(i,:) = beta0';
        draws.taul(i,:) = [taul  taul_d];
        draws.phil(i,:) = [phil'  phil_d'];
%         draws.count_beta0 = draws.count_beta0 + count_beta0; 
%         draws.logrw_beta0(i) = logrw_beta0;%beta0        

        draws.count_pz = draws.count_pz + count_pz; 
        draws.logrw(i,:) = logrw';
        for j = 1:K
            draws.pz{j}(i,:) = pz(:,j)';
            draws.ind2{j}(i,:) = (v(j)^2)*ind2(:,j)';
        end %z        
        
        draws.yfit(i,:) = yfit';                
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end        
        
        if ind_sparse == 1
            draws.v_sparse(i,:) = v_sparse';
            draws.beta0_sparse(i,:) = beta0_sparse';
            for j = 1:K
                draws.beta_sparse{j}(i,:) = beta_sparse(:,j)';
            end
        end
        
        if ind_forecast == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
            if ind_sparse == 1
                draws.bn_smean(i,:) = bn_smean';
                draws.bn_scov{i} = bn_scov; 
            end            
        end        
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end
draws.count_pz = draws.count_pz / ndraws;
% draws.count_v = draws.count_v / ndraws;
% draws.count_beta0 = draws.count_beta0 / ndraws;
draws.count_q = draws.count_q / ndraws;
draws.count_r = draws.count_r / ndraws;


    





