% Estimate a RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt = b_{j,t-1} + N(0,d_jt), j = 1, ..., K
% d_jt = vj2 * I(z_jt > c_j),
% z_jt = (1-rho_j) * u_j + rho_j * z_{j,t-1} + N(0,sj),
%
% use GCK algorithm for zt (adpative MH， multivariate)
% draw static parameters by integrating out bt (adaptive MH, multivariate)
% use ASIS for extra boosting


clear;
dbstop if warning;
dbstop if error;
rng(3721);


mdl = {'CV', 'MI', 'KHS','LT','NWest'};
for mdlj = 1:5
    disp(mdl{mdlj});



    %% Read data to get y and x
    read_file = 'Industrial_Output_Github.xlsx';
    read_sheet = 'Data';   
    data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:E272');  
    ng = size(data,1);    
    y = data(5:ng,1); 
    x = [ones(ng-4,1) data(4:(ng-1),1) data(3:(ng-2),1) data(2:(ng-3),1) data(1:(ng-4),1) ...
        data(4:(ng-1),4) data(4:(ng-1),3)]; %AR4 + Short_Chg + Term_SPD (1 lag)
    


    
    
    %% Set the size of estimation/prediction sample
    [n,nx] = size(x);
    npred = 40; %number of predictions >= 0
    nest = n - npred; %number of estimation data
    disp(['nobs = ', num2str(n), ', nx = ', num2str(nx)]);
    disp(['nest = ', num2str(nest), ', npred = ', num2str(npred)]); 
    
    
    %% Configuration
    ind_SV = 1; %if SV for measurement noise variance
    ind_sparse = 0; %if sparsifying is needed (fix at 0)
    disp(['SV = ', num2str(ind_SV),', sparse = ', num2str(ind_sparse)]);    



    %% MCMC
    ndraws = 5000*3;
    burnin = 5000;    
    disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);

    tic;
    if npred == 0 %in-sample estimation only 
        ind_forecast = 0;   
        yest = y;
        xest = [ones(n,1)  normalize_data(x(:,2:nx))];
        switch mdlj          
            case 1 %triple gamma
                draws = RWTVP(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
            case 2 %mixture innovation
%                 draws = RWTVP_MI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                MI_scenarios = [zeros(1,nx); [1 zeros(1,nx-1)]; ones(1,nx)];
                draws = RWTVP_RMI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast, MI_scenarios);  
            case 3 %dynamic horseshoe
                draws = RWTVP_KHS3_scale(yest, xest, burnin, ndraws, ind_SV, ind_forecast);
            case 4 %LTVP
                draws = RWTVP_LinkConst_AA_same(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
            case 5 %latent threshold
                draws = RWTVP_NWest6(yest, xest, burnin, ndraws, ind_SV);
            otherwise
                error('Wrong model');
        end
        disp([mdl{mdlj},' is completed!']);
        save(['Est_',mdl{mdlj},'_industrial_output', '.mat'], 'draws');
        toc;
    else
        ind_forecast = 1;
        logpredlike = zeros(npred,2);
        prederror = zeros(npred,2);
        valid_percent = zeros(npred,2); %count conditional likelihoods that are not NaN or Inf     
        for predi = 1:npred 
            % process data
            nesti = nest + predi - 1;
            yi = y(1:nesti,:);
            xi = x(1:nesti,:); %rescaling x is possible 
            
            yest = yi;
            xest = [ones(nesti,1)  normalize_data(xi(:,2:nx))];

            % estimate the model
            switch mdlj          
                case 1 %triple gamma
                    draws = RWTVP(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                case 2 %mixture innovation
    %                 draws = RWTVP_MI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                    MI_scenarios = [zeros(1,nx); [1 zeros(1,nx-1)]; ones(1,nx)];
                    draws = RWTVP_RMI(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast, MI_scenarios);  
                case 3 %dynamic horseshoe
                    draws = RWTVP_KHS3_scale(yest, xest, burnin, ndraws, ind_SV, ind_forecast);
                case 4 %LTVP
                    draws = RWTVP_LinkConst_AA_same(yest, xest, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);
                case 5 %latent threshold
                    draws = RWTVP_NWest6(yest, xest, burnin, ndraws, ind_SV);
                otherwise
                    error('Wrong model');
            end
            
            % prediction
            xtp1 = x(nesti+1,:)'; 
            ytp1 = y(nesti+1);
            
            xmean = mean(xi(:,2:nx))';
            xstd = std(xi(:,2:nx))';
            xtp1_normalized = [1; (xtp1(2:nx) - xmean)./xstd];
            
            if mdlj==1 %triple gamma
                ind_homo = 1;
                ind_KHS = 0;
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_TVP(draws,...
                    xtp1_normalized, ytp1, ind_SV, ind_homo, ind_KHS); 
            elseif mdlj == 2 %mixture innovation
%                 [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_MI(draws,...
%                     xtp1_normalized, ytp1, ind_SV); 
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_RMI(draws,...
                    xtp1_normalized, ytp1, ind_SV, MI_scenarios);                
            elseif mdlj==3 %dynamic horseshoe 
                ind_KHS = 1;
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_TVP_HS(draws,...
                    xtp1_normalized, ytp1, ind_SV, ind_KHS);
            elseif mdlj == 4 %LTVP
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_LinkConst(draws,...
                    xtp1_normalized, ytp1, ind_SV);    
            else %latent threshold
                [ytp1_pdf, ytp1_mean, ytp1_var, ytp1_pdf_vec, ind_valid] = pred_NWest6(draws,...
                    xtp1_normalized, ytp1, ind_SV);             
            end

            % store log likelihoods and prediction error
            logpredlike(predi,1) = log(ytp1_pdf(1))';
            prederror(predi,1) = ytp1 - mean(ytp1_mean(:,1));
            valid_percent(predi,1) = sum(ind_valid(:,1))/ndraws;
           
            disp(['Prediction ', num2str(predi), ' out of ', num2str(npred), ' is finished!']);
            toc;   
            disp(' ');              
        end %end of prediction loop      
    end %end of prediction choice
end %end of model loop






