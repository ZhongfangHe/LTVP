% Simulate data

clear;
rng(1234567);


%% Simulate
mu = [0  1]'; %benchmark is -1(R2=0.88); now R2 = 0.54 (1), 0.13 (3)
% mu = 2;
nrep = length(mu);
n = 300;
K = 6;
for j = 1:nrep
    % generate x
    x = randn(n,K);

    % generate beta
    btrue = zeros(n,K);
    if K > 1
        btrue(:,1) = cumsum(0.1*randn(n,1)); %coef1: RW
        
        bp = [round(n/3)  round(2*n/3)];
        btrue(bp(1):bp(2),2) = 1;
        btrue(bp(2)+1:n,2) = -0.5; %coef2: 2 chang points        
        
        bp = round(n/2);
        btrue(bp+1:n,3) = 1; %coef3: 1 chang point
        
        btrue(:,4) = ones(n,1); %coef4: ones

    else
        bp = [round(n/3)  round(2*n/3)];
        btrue(bp(1):bp(2)) = 1;
        btrue(bp(2)+1:n) = -1;
    end
    
    % generate resid variance
    yfit = sum(btrue .* x, 2);
    sig2true = exp(mu(j))*ones(n,1); %different R2 by adjusting mu
    for t = 2:n
        logs = (1-0.9)*mu(j) + 0.9*log(sig2true(t-1)) + 0.1*randn;
        sig2true(t) = exp(logs);
    end
%     sig2true = mu(j)*ones(n,1);
    sigtrue = sqrt(sig2true);

    
    % generate y with different noise level, write output
    y = yfit + sigtrue.* randn(n,1);
    R2 = var(yfit)/var(y);
    disp(['R2 = ', num2str(R2)]);
    
    write_file = 'Simulated_Data_SV_diffR2.xlsx';
    write_sheet = ['D',num2str(j)];
    title = cell(1,2*K+2);
    title{1} = ['y(R2=', num2str(R2), ')'];
    title{2} = 'sig';
    for jj = 1:K
        title{jj+2} = ['x',num2str(jj)];
        switch jj
            case 1
                title{K+2+jj} = 'RW';
            case 2
                title{K+2+jj} = 'CP2';
            case 3
                title{K+2+jj} = 'CP1';
            case 4
                title{K+2+jj} = 'One';                   
            otherwise
                title{K+2+jj} = 'Zero';
        end
    end
    writecell(title, write_file, 'Sheet', write_sheet, 'Range', 'A1');
    writematrix([y sigtrue x  btrue], write_file, 'Sheet', write_sheet, 'Range', 'A2');
end


