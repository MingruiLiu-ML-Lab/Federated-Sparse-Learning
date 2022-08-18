function eval_list = LassoRun()

    K = 10;
    m_per_round = 10;
    R  = 2000; % round number
    bs = 10; % batch size
    plot_path = 'lasso_results.eps';
    results_path = 'lasso_results.mat';

    % data generation process
    rng('default');
    d = 1024;
    m = 64; % numeber of clients
    nm = 128; % pairs of observations
    w_real=zeros(d,1);
    %b_real = mvnrnd(0,1);
    b_real = 0;
    for i=1:512
        w_real(i)=1;
    end
    mu = mvnrnd(zeros(d,1),eye(d), m);
    Sigma = zeros(d);
    for i = 1:d
        for j = 1:d
            Sigma(i, j) = 0.5^(abs(i - j));
        end
    end

    delta = mvnrnd(zeros(d,1), Sigma, m*nm);
    epsilon = mvnrnd(0, 1, nm*m);
    mu = mu';
    delta=delta';
    X = zeros(d, nm*m);
    for i=1:m
        X(:,nm*(i-1)+1:nm*i) = repmat(mu(:,i),1,nm)+delta(:,nm*(i-1)+1:nm*i);
    end
    Y = zeros(1, nm*m);
    Y = X'*w_real + epsilon; 

    eval_list = cell(5, 1);
    lambda = 0.5^5; % regularization parameter

    % same initial point for 5 methods
    ini_x0 = randn(d,1);

    % learning rate for FedDA, FedMid
    etac = 0.001; % client learning rate
    etas = 1; % server learning rate

    %% FedAvg
    % wr = zeros(d,1);
    % W = zeros(d, m);
    % Delta = zeros(d, 1);
    % wtemp = zeros(d, 1);
    % res_fedavg = zeros(R, 5);

    % for r=1:R
    %     W = repmat(wr, 1, m);
    %     wtemp = wr; % record the snapshot
    %     for i = 1:m
    %         for k=1:K
    %             idx = datasample(nm*(i-1)+1:nm*i, bs, 'Replace', false); % sample minibatch size 1
    %             g = loss_grad(W(:, l), X(:,idx), Y(idx)); 
    %             g = g + lambda * sign(wtemp);
    %             W(:, l) = W(:, l) - etac*g; % client dual update
    %         end
    %     end
    %     Delta = mean(W - wtemp, 2); % correspond to delta_r 
    %     wr = wr + etas*Delta; % server dual update;
    %     res_fedavg(r, :) = eval_metric_lasso(wr, w_real, X, Y);
    %     if mod(r, 100) == 0
    %         disp(norm(wr - w_real))
    %     end
    % end

    %% FedMirror
    wr = ini_x0;
    Delta = zeros(d, 1);
    wtemp = zeros(d, 1);
    res_FedMid = zeros(R, 4);

    for r=1:R
        wtemp = wr; % record the snapshot
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        W = repmat(wr, 1, m_per_round);
        l = 0;
        for i = id_client
            l = l + 1;
            for k=1:K
                idx = datasample(nm*(i-1)+1:nm*i, bs, 'Replace', false); % sample minibatch
                g = loss_grad(W(:, l), X(:,idx), Y(idx)); 
                W(:, l) = l1_soft(W(:, l) - etac*g, etac*lambda); % client dual update
            end
        end
        Delta = mean(W - wtemp, 2); % correspond to delta_r 
        wr = l1_soft(wr + etas*Delta, etas*etac*lambda); % server dual update;
        res_FedMid(r, :) = eval_metric_lasso(wr, w_real, X, Y);
        if mod(r, 100) == 0
            disp(norm(wr - w_real))
        end
    end
    eval_list{1} = res_FedMid;

    %% FedDualAvg
    z = ini_x0;
    Delta = zeros(d, 1);
    Ztemp = zeros(d,m);
    wrk=zeros(d,1);
    res_FedDA = zeros(R, 4);

    for r=1:R
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        Z=repmat(z, 1, m_per_round);
        Ztemp = Z; % record the snapshot
        l = 0;
        for i = id_client
            l = l+1;
            for k=1:K
                etark = etas * etac*r*K+etac*k;
                wtemp = l1_soft(Z(:, l),etark*lambda); % proximal mapping, retrieve primal
                idx = datasample(nm*(i-1)+1:nm*i,bs, 'Replace',false); % sample minibatch size 1
                g = loss_grad(wtemp,X(:,idx), Y(idx)); 
                Z(:, l) = Z(:, l)-etac*g; % client dual update
            end
        end
        Delta = mean(Z-Ztemp, 2); % correspond to delta_r 
        z = z+etas*Delta; % server dual update;
        wr = l1_soft(z, etas*etac*(r+1)*K*lambda);
        res_FedDA(r, :) = eval_metric_lasso(wr, w_real, X, Y);
        if mod(r, 100) == 0
            disp(norm(wr-w_real))
        end
    end
    eval_list{2} = res_FedDA;

    % Fast FedDualAvg with strong convexity (Our proposal)
    mu = 0.1;
    gamma = 550;
    x0 = ini_x0;
    xr = x0; % cumulative primal variable
    wr = x0; % primal
    gr = zeros(d, 1); % cumulative gradient of server
    res_FastFedDA = zeros(R, 4);

    for r = 1:R
        cr = lambda;
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        gr_client = repmat(gr, 1, m_per_round);
        xr_client = repmat(xr, 1, m_per_round);
        l = 0;
        for i = id_client
            wr_i = wr; %same starting point
            l = l+1;
            for k = 1:K
                idx = datasample(nm*(i-1)+1:nm*i, bs, 'Replace', false); % sample minbatch
                G_i = loss_grad(wr_i, X(:,idx), Y(idx)); % compute gradients
                gr_client(:, l) = gr_client(:, l) + G_i; % cumulative gradient of cilent

                eta_k = (r-1)*K + k;
                ar_i = gr_client(:, l)/eta_k - 0.5*mu * xr_client(:, l)/eta_k - gamma * x0/eta_k; %parameter in client's optimization
                br_i = 0.5*mu + gamma/eta_k;
                wr_i = l1_soft(-ar_i/br_i, cr/br_i); % primal update of client
                xr_client(:, l) = xr_client(:, l) + wr_i;
            end
        end
        eta_r = r*K;
        gr = mean(gr_client, 2); % aggregate gradients
        xr = mean(xr_client, 2);

        ar = gr/eta_r - 0.5*mu * xr/eta_r - gamma * x0/eta_r; % parameter in server's optimization
        br = 0.5*mu + gamma/eta_r;
        wr = l1_soft(-ar/br, cr/br); % primal update of client
        res_FastFedDA(r, :) = eval_metric_lasso(wr, w_real, X, Y);
        if mod(r, 100) == 0
            disp(norm(wr-w_real))
        end
        xr = xr + wr;
    end
    eval_list{3} = res_FastFedDA;

    % FedDualAvg with strong convexity
    mu = 0.1;
    gamma = 600;
    xr = ini_x0; % cumulative primal
    wr = xr; % primal
    gr = zeros(d, 1); % gradient
    res_AFedDA = zeros(R, 4);

    for r = 1:R
        br = 0.5*mu + gamma/(r*K);
        cr = lambda;
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        gr_client = repmat(gr, 1, m_per_round);
        l = 0;
        for i = id_client
            l = l+1;
            wr_i = wr; %same starting point
            for k = 1:K
                idx = datasample(nm*(i-1)+1:nm*i, bs, 'Replace', false); % sample minbatch
                G_i = loss_grad(wr_i, X(:,idx), Y(idx)); % compute gradients
                gr_client(:, l) = gr_client(:, l) + G_i; % cumulative gradient of cilent

                ar_i = gr_client(:, l)/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); %parameter in client's optimization
                wr_i = l1_soft(-ar_i/br, cr/br); % primal update of client
            end
        end
        gr = mean(gr_client, 2); % aggregate gradients
        ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
        wr = l1_soft(-ar/br, cr/br); % primal update of client
        res_AFedDA(r, :) = eval_metric_lasso(wr, w_real, X, Y);
        if mod(r, 100) == 0
            disp(norm(wr-w_real))
        end
        xr = xr + wr;
    end
    eval_list{4} = res_AFedDA;

    % Multi-stage FedDualAvg with strong convexity
    mu = 0.1;
    gamma = 600;
    x0 = ini_x0;
    xr = x0; % cumulative primal
    wr = x0; % primal
    gr = zeros(d, 1); % gradient
    S = 3;
    res_MFedDA = zeros(R, 4);
    lambda_set = [0.5^3 0.5^4 0.5^5];
    rho_set = [1e4*(0.5) 1e4*(0.5)^2 1e4*(0.5)^3];
    R_set = [250 250 1500];
    r_now = 0;

    for s = 1:S
        x0 = wr; % re-start
        xr = wr;
        gr = zeros(d, 1);
        rho = rho_set(s); % update radius of L1 ball
        lambda_s = lambda_set(s); % update lambda value
        for r = 1:R_set(s)
            br = 0.5*mu + gamma/(r*K);
            cr = lambda_s;
            id_client = datasample(1:m, m_per_round, 'Replace', false);
            gr_client = repmat(gr, 1, m_per_round);
            l = 0;
            for i = id_client
                l = l + 1;
                wr_i = wr; %same starting point
                for k = 1:K
                    idx = datasample(nm*(i-1)+1:nm*i, bs, 'Replace', false); % sample minbatch
                    G_i = loss_grad(wr_i, X(:,idx), Y(idx)); % compute gradients
                    gr_client(:, l) = gr_client(:, l) + G_i; % cumulative gradient of cilent
        
                    ar_i = gr_client(:, l)/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); %parameter in client's optimization
                    wr_i = l1_soft_cs(-ar_i/br, cr/br, x0, rho); % primal update of client
                end
            end
            gr = mean(gr_client, 2); % aggregate gradients
            ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
            wr = l1_soft_cs(-ar/br, cr/br, x0, rho); % primal update of client
            r_now = r_now + 1;
            res_MFedDA(r_now, :) = eval_metric_lasso(wr, w_real, X, Y);
            if mod(r, 100) == 0
                disp(norm(wr-w_real))
            end
            xr = xr + wr;
        end
    end
    eval_list{5} = res_MFedDA;

    % save and plot results
    plot_lasso(eval_list, plot_path);
    results = struct('eval_list', eval_list);
    save(results_path, 'results');
