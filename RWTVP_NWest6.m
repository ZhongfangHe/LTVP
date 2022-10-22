% Estimate a Nakajima-West model:
% yt = xt' * (bt .* I(|bt|>v)) + N(0,sig2t), 
% b_jt = (1-rho_j) * u_j + rho_j * b_{j,t-1} + N(0,a2j),
%
% ASIS for bt
% normalize threshold


function draws = RWTVP_NWest6(y, x, burnin, ndraws, ind_SV)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws after burnin
%   ind_SV: an indicator if SV for measurement noise variance
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
% minNum = 1e-100;
% maxNum = 1e100;


%% Priors: threshold of bt, 
psiv = 1*ones(K,1);
v = gamrnd(0.5,2*psiv);
v2 = v.^2;


%% Priors: long-run mean for bt, u ~ N(0, diag(psiu))
psiu = 1*ones(K,1);
u = sqrt(psiu./v2) .* randn(K,1);


%% Priors: slope of bt, rho ~ 2*beta-1
r1 = 20; r2 = 1.5;
rho = 2*betarnd(r1,r2,K,1)-1;
rho2 = rho.^2;


%% Priors: conditional var of bt, a2 ~ IG
psia = 1*ones(K,1);
a = sqrt(psia./v2).*randn(K,1);
a2 = a.^2;



%% Initialize latent index
pz = zeros(n,K);



%% Set up adaptive MH
pstar = 0.25;
tmp_const = -norminv(0.5*pstar);
AMH_c = 1/(K * pstar * (1-pstar)) + (1-1/K)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;

logrw = zeros(n,1);
logrw_start = logrw;
drawi_start = zeros(n,1); %z

logrw_a = 0;
logrw_start_a = logrw_a;
drawi_start_a = 0; %a

logrw_u = 0;
logrw_start_u = logrw_u;
drawi_start_u = 0; %u


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
draws.a = zeros(ndraws,K);
draws.count_a = 0;
draws.logrw_a = zeros(ndraws,1); %a

draws.u = zeros(ndraws,K);
draws.count_u = 0;
draws.logrw_u = zeros(ndraws,1); %u

draws.v = zeros(ndraws,K);
draws.count_v = zeros(K,1); %v

draws.rho = zeros(ndraws,K);
draws.count_rho = zeros(K,1); %rho

draws.beta = cell(K,1);
draws.beta_sparse = cell(K,1);
for j = 1:K
    draws.beta{j} = zeros(ndraws,n);
    draws.beta_sparse{j} = zeros(ndraws,n);
end %beta

draws.count_pz = zeros(n,1);
draws.logrw = zeros(ndraws,n);
draws.pz = cell(K,1);
for j = 1:K
    draws.pz{j} = zeros(ndraws,n);
end %z

if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

tic;
ntotal = burnin + ndraws;
pz_mean = zeros(n,K);
pz_cov = cell(n,1);
for t = 1:n
    pz_cov{t} = zeros(K,K);
end
a_mean = zeros(K,1);
a_cov = zeros(K,K);
u_mean = zeros(K,1);
u_cov = zeros(K,K);
for drawi = 1:ntotal   
    % Draw z (normalized beta)    
    pz_mean_old = pz_mean;
    pz_cov_old = pz_cov;
    pz_old = pz;
    logrw_old = logrw; 
    [pz, logrw, count_pz, pz_mean, pz_cov] = pz_simulator_NWest6(y, x, vary,...
        u, rho, a, v, pz_old, pstar, AMH_c, ...
        logrw_old, logrw_start, drawi_start, drawi, burnin, pz_mean_old, pz_cov_old);    

    
    % Draw u (MH)
    count_u = 0;

    u_old = u;
    if drawi < 100
        A = eye(K);
    else  
        A = u_cov + 1e-6 * eye(K) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(K,1),A)'; %correlated normal
    u_new = u_old + exp(logrw_u) * eps; 
    
    u2_old = u_old.^2;
    u2_new = u_new.^2;
    logprior_old = -0.5 * sum(v2.*u2_old./psiu);
    logprior_new = -0.5 * sum(v2.*u2_new./psiu); %p(u)

    xv = x.*repmat(v',n,1);
    tmp = repmat(a',n,1).*pz; 
    b = repmat(u_old',n,1) + tmp;
    m = sum(xv.*b.*double((abs(b)-1)>0),2);
    loglike_old = -0.5*sum(((y-m).^2)./vary);
    b = repmat(u_new',n,1) + tmp;
    m = sum(xv.*b.*double((abs(b)-1)>0),2);
    loglike_new = -0.5*sum(((y-m).^2)./vary); %p(y|z,u)

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        u = u_new;
        if drawi > burnin
            count_u = 1;
        end
    end
     

    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start_u/K);
    d = max(ei - ei_start, 20);
    logrwj = logrw_u + AMH_c * (p - pstar)/d;
    if abs(logrwj - logrw_start_u) > 1.0986 %log(3) ~= 1.0986 
        drawi_start_u = drawi;
        logrw_start_u = logrw_u;
    end %restart when useful to allow for larger movement    
    logrw_u = logrwj; %update proposal stdev
    
    
    u_mean_old = u_mean;
    u_cov_old = u_cov;
    u_mean = (u_mean_old * (drawi-1) + u) / drawi;
    u_cov = (drawi - 1) * (u_cov_old + u_mean_old * u_mean_old') / drawi + ...
        u * u' / drawi - u_mean * u_mean'; %update the sample covariance


    % Draw a (MH)
    count_a = 0;

    a_old = a;
    if drawi < 100
        A = eye(K);
    else  
        A = a_cov + 1e-6 * eye(K) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(K,1),A)'; %correlated normal
    a_new = a_old + exp(logrw_a) * eps; 
    
    a2_old = a_old.^2;
    a2_new = a_new.^2;
    logprior_old = -0.5 * sum(v2.*a2_old./psia);
    logprior_new = -0.5 * sum(v2.*a2_new./psia); %p(a)

    tmp = repmat(u',n,1);  
    b = repmat(a_old',n,1).*pz + tmp;
    m = sum(xv.*b.*double((abs(b)-1)>0),2);
    loglike_old = -0.5*sum(((y-m).^2)./vary);
    b = repmat(a_new',n,1).*pz + tmp; 
    m = sum(xv.*b.*double((abs(b)-1)>0),2);
    loglike_new = -0.5*sum(((y-m).^2)./vary);

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        a = a_new;
        a2 = a2_new;
        if drawi > burnin
            count_a = 1;
        end
    end
     

    p = exp(min(0,logprob));
    ei = max(200, drawi/K);
    ei_start = max(200, drawi_start_a/K);
    d = max(ei - ei_start, 20);
    logrwj = logrw_a + AMH_c * (p - pstar)/d;
    if abs(logrwj - logrw_start_a) > 1.0986 %log(3) ~= 1.0986 
        drawi_start_a = drawi;
        logrw_start_a = logrw_a;
    end %restart when useful to allow for larger movement    
    logrw_a = logrwj; %update proposal stdev
    
    
    a_mean_old = a_mean;
    a_cov_old = a_cov;
    a_mean = (a_mean_old * (drawi-1) + a) / drawi;
    a_cov = (drawi - 1) * (a_cov_old + a_mean_old * a_mean_old') / drawi + ...
        a * a' / drawi - a_mean * a_mean'; %update the sample covariance    
    


     % Asis: compute beta_star (SA)
    beta = repmat(u',n,1) + repmat(a',n,1).*pz;

    % Asis: update u
    for j = 1:K
        uA_inv = v2(j)/psiu(j) + (n+(n-2)*rho2(j)-2*rho(j)*(n-1)) / a2(j);
        uA = 1/uA_inv;
        tmp = (1-rho2(j))*beta(1,j)+(1-rho(j))*sum(beta(2:n,j)-rho(j)*beta(1:n-1,j));
        ua = uA * tmp / a2(j);
        u(j) = ua + sqrt(uA) * randn;
    end
    
    % Asis: update a2
    resid1 = sqrt(1-rho2)' .* (beta(1,:)-u');
    resid2 = beta(2:n,:) - repmat((1-rho').*u',n-1,1) - repmat(rho',n-1,1).*beta(1:n-1,:);
    resid = [resid1;resid2];
    resids = resid.^2;
    a_sign = sign(a);
    for j = 1:K 
        [a2(j),~] = gigrnd(0.5-0.5*n, v2(j)/psia(j), sum(resids(:,j)), 1);
    end
    a = sqrt(a2) .* a_sign; 

    
    % Asis: compute back pz 
    pz = (beta - repmat(u',n,1))./repmat(a',n,1);
    

    % Draw rho (independent MH)
    count_rho = zeros(K,1);
    for j = 1:K 
        pzj = pz(:,j);
        
        rhoj_old = rho(j);
        rho2j_old = rho2(j); 

        prop_var_inv = sum(pzj(1:n-1).^2);
        prop_var = 1/prop_var_inv;
        prop_mean = prop_var*sum(pzj(2:n).*pzj(1:n-1));
        rhoj_new = prop_mean + sqrt(prop_var)*randn;
        rho2j_new = rhoj_new.^2;
        
        if abs(rhoj_new)<(1-1e-10)
            logprior_old = (r1-1)*log(1+rhoj_old) + (r2-1)*log(1-rhoj_old);
            logprior_new = (r1-1)*log(1+rhoj_new) + (r2-1)*log(1-rhoj_new);%p(rho)
 
            tmp = pzj(1)^2; 
            loglike1_old = 0.5*log(1-rho2j_old) - 0.5*(1-rho2j_old)*tmp;
            loglike1_new = 0.5*log(1-rho2j_new) - 0.5*(1-rho2j_new)*tmp; %p(z|rho)  

            logprob = logprior_new + loglike1_new ...
                - logprior_old - loglike1_old;
            if log(rand) <= logprob
                rho(j) = rhoj_new;
                rho2(j) = rho2j_new;
                if drawi > burnin
                    count_rho(j) = 1;
                end
            end
        end
    end
    
    
    
    % Draw v (MH) 
    count_v = zeros(K,1);
    b = repmat(a',n,1).*pz + repmat(u',n,1);
    xb = x.*b.*double((abs(b)-1)>0);
    for j = 1:K
        vj_old = v(j);
        v2j_old = v2(j);
        
        if K > 1
            idx = setdiff(1:K,j);
            prop_y = y - xb(:,idx) * v(idx);
        else
            prop_y = y;
        end
        prop_x = xb(:,j);
        prop_var_inv = sum((prop_x.^2)./vary);
        prop_var = 1/prop_var_inv;
        prop_mean = prop_var * sum(prop_x.*prop_y./vary);
        vj_new = prop_mean + sqrt(prop_var)*randn;
        v2j_new = vj_new^2;
        
        if vj_new > 0
            logprior_old = -0.5*log(vj_old)-0.5*vj_old/psiv(j);
            logprior_new = -0.5*log(vj_new)-0.5*vj_new/psiv(j); %p(v2)
            
            loglike1_old = log(vj_old)-0.5*v2j_old*u(j)*u(j)/psiu(j);
            loglike1_new = log(vj_new)-0.5*v2j_new*u(j)*u(j)/psiu(j); %p(u|v2)
            
            loglike2_old = log(vj_old)-0.5*v2j_old*a2(j)/psia(j);
            loglike2_new = log(vj_new)-0.5*v2j_new*a2(j)/psia(j); %p(a2|v2)
            
            logprob = logprior_new + loglike1_new + loglike2_new ...
                - logprior_old - loglike1_old - loglike2_old;
            if log(rand) <= logprob
                v(j) = vj_new;
                v2(j) = v2j_new;
                if drawi > burnin
                    count_v(j) = 1;
                end
            end                
        end %if vj_new>0
    end %loop of j   
    

       
    
    
    % Residual variance
    beta = repmat(u',n,1) + repmat(a',n,1).*pz;
    yfit = sum(x .* repmat(v',n,1).*beta .* double(abs(beta)>1),2);
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
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K
            draws.beta{j}(i,:) = beta(:,j)'*v(j);
        end  
        
        draws.a(i,:) = a'.*v';       
        draws.count_a = draws.count_a + count_a; 
        draws.logrw_a(i) = logrw_a;%a  
        
        draws.u(i,:) = u'.*v';
        draws.logrw_u(i) = logrw_u;
        draws.count_u = draws.count_u + count_u; %u
        
        draws.rho(i,:) = rho';
        draws.count_rho = draws.count_rho + count_rho; %rho
        
        draws.v(i,:) = v';       
        draws.count_v = draws.count_v + count_v; %v              

        draws.count_pz = draws.count_pz + count_pz; 
        draws.logrw(i,:) = logrw';
        for j = 1:K
            draws.pz{j}(i,:) = pz(:,j)';
            draws.beta_sparse{j}(i,:) = (v(j)*beta(:,j).*double(abs(beta(:,j))>1))'; 
        end %z        
        
        draws.yfit(i,:) = yfit';                
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end              
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end
draws.count_pz = draws.count_pz / ndraws;
draws.count_v = draws.count_v / ndraws;
draws.count_a = draws.count_a / ndraws;
draws.count_u = draws.count_u / ndraws;
draws.count_rho = draws.count_rho / ndraws;



