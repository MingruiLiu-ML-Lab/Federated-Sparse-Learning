function eval_list = NuclearRun()

    K = 10;
    m_per_round = 10;
    R = 1500; % round number
    bs = 10; %batch size
    plot_path = 'nuclear_results.eps';
    results_path = 'nuclear_results.mat';

    %% data generation process
    rank_real = 16; % ground truth rank
    d = 32; % dimension of W
    m = 64; % number of clients
    nm = 128; % sample size in each client
    lambda = 0.1;

    W_real = [eye(rank_real) zeros(rank_real,d-rank_real); zeros(d-rank_real, rank_real) zeros(d-rank_real)]; %ground truth W
    Mu_set = cell(m,1); % mu for each client
    for i = 1:m
        Mu_set{i} = randn(d);
    end

    Data_set = cell(m, 1); % data set in each client
    for i = 1:m
        local_data = cell(nm, 2);
        for j = 1:nm
            X = randn(d) + Mu_set{m};
            local_data{j, 1} = X;
            local_data{j ,2} = trace(X'*W_real) + randn(1);
        end
        Data_set{i} = local_data;
    end

    % same initial point for 5 methods
    ini_x0 = randn(d);
    
    % learning rate for FedMiD and FedDA
    etac = 0.001; % client learning rate
    etas = 1; % server learning rate

    %% FedMirror
    wr = ini_x0;
    W = zeros(d);
    Delta = zeros(d);
    Wtemp = zeros(d);
    res_FedMid = zeros(R, 4);

    for r=1:R
        Wtemp = wr; % record the snapshot
        Wnew = zeros(d);
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        for i = id_client
            w_client = wr;
            for k=1:K
                g = nuc_grad(w_client, bs, Data_set{i});
                w_client = nuc_prox(w_client - etac*g, etac*lambda); % client dual update
            end
            Wnew = Wnew + w_client;
        end
        Delta = Wnew/m_per_round - Wtemp; % correspond to delta_r 
        wr = nuc_prox(wr + etas*Delta, etas*etac*lambda); % server dual update;
        res_FedMid(r, :) = eval_metric_nuc(wr, W_real, Data_set);
        if mod(r, 100) == 0
            disp(norm(wr - W_real, 'fro'))
        end
    end

    %% FedDualAvg
    z = ini_x0;
    Delta = zeros(d);
    Ztemp = zeros(d);
    res_FedDA = zeros(R, 4);

    for r=1:R
        Ztemp = z; % record the snapshot
        Znew = zeros(d);
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        for i = id_client
            z_client = z;
            for k=1:K
                etark = etas * etac*r*K + etac*k;
                wtemp = nuc_prox(z_client, etark*lambda); % proximal mapping, retrieve primal
                
                g = nuc_grad(wtemp, bs, Data_set{i}); % compute gradient
                z_client = z_client - etac*g; % client dual update
            end
            Znew = Znew + z_client;
        end
        Delta = Znew/m_per_round - Ztemp; % correspond to delta_r 
        z = z + etas*Delta; % server dual update;
        wr = nuc_prox(z, etas*etac*(r+1)*K*lambda);
        res_FedDA(r, :) = eval_metric_nuc(wr, W_real, Data_set);
        if mod(r, 100) == 0
            disp(norm(wr-W_real, 'fro'))
        end
    end


    %% Fast FedDualAvg with strong convexity
    mu = 0.1;
    gamma = 550;
    x0 = ini_x0;
    xr = x0; % cumulative primal variable
    wr = x0; % primal
    gr = zeros(d); % cumulative gradient of server
    res_FastFedDA = zeros(R, 4);

    for r = 1:R
        cr = lambda;
        gr_new = zeros(d);
        wr_new = zeros(d);
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        for i = id_client
            gr_client = gr;
            wr_client = xr;
            wr_i = wr; %same starting point
            for k = 1:K
                eta_k = (r-1)*K + k;
                ar_i = gr_client/eta_k - 0.5*mu * wr_client/eta_k - gamma * x0/eta_k; %parameter in client's optimization
                br_i = 0.5*mu + gamma/eta_k;
                wr_i = nuc_prox(-ar_i/br_i, cr/br_i); % primal update of client

                G_i = nuc_grad(wr_i, bs, Data_set{i}); % compute gradients
                gr_client = gr_client + G_i; % cumulative gradient of cilent
                wr_client = wr_client + wr_i;
            end
            wr_new = wr_new + wr_client;
            gr_new = gr_new + gr_client;
        end
        eta_r = r*K;
        gr = gr_new/m_per_round; % aggregate gradients
        xr = wr_new/m_per_round;

        ar = gr/eta_r - 0.5*mu * xr/eta_r - gamma * x0/eta_r; % parameter in server's optimization
        br = 0.5*mu + gamma/eta_r;
        wr = nuc_prox(-ar/br, cr/br); % primal update of client
        res_FastFedDA(r, :) = eval_metric_nuc(wr, W_real, Data_set);
        if mod(r, 100) == 0
            disp(norm(wr-W_real, 'fro'))
        end
        xr = xr + wr;
    end

    %% Single-stage FedDualAvg with strong convexity
    mu = 0.1;
    gamma = 600;
    x0 = ini_x0;
    xr = x0; % cumulative primal
    wr = x0; % primal
    gr = zeros(d); % gradient
    cr = lambda;
    res_SFedDA = zeros(R, 4);


    for r = 1:R
        br = 0.5*mu + gamma/(r*K);
        gr_new = zeros(d);
        id_client = datasample(1:m, m_per_round, 'Replace', false);
        for i = id_client
            gr_client = gr;
            wr_i = wr; %same starting point
            for k = 1:K
                ar_i = gr_client/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); %parameter in client's optimization
                wr_i = nuc_prox(-ar_i/br, cr/br); % primal update of client

                G_i = nuc_grad(wr_i, bs, Data_set{i}); % compute gradients
                gr_client = gr_client + G_i; % cumulative gradient of cilent
            end
            gr_new = gr_new + gr_client;
        end
        gr = gr_new/m_per_round; % aggregate gradients
        ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
        wr = nuc_prox(-ar/br, cr/br); % primal update of client
        res_SFedDA(r, :) = eval_metric_nuc(wr, W_real, Data_set);
        if mod(r, 100) == 0
            disp(norm(wr - W_real, 'fro'))
        end
        xr = xr + wr;
    end

    % Multi-stage FedDualAvg with strong convexity
    mu = 0.1;
    gamma = 600;
    x0 = ini_x0;
    xr = x0; % cumulative primal
    wr = x0; % primal
    gr = zeros(d); % gradient
    S = 3;
    R0 = R/S;
    res_MFedDA = zeros(R, 4);
    lambda_set = [0.3 0.15 0.1];
    rho_set = [1e4*(0.5) 1e4*(0.5)^2 1e4*(0.5)^3];
    R_set = [250 250 1000];
    r_now = 0;

    for s = 1:S
        x0 = wr; % re-start
        xr = wr;
        gr = zeros(d, 1);
        rho = rho_set(s);
        lambda_s = lambda_set(s);
        for r = 1:R_set(s)
            br = 0.5*mu + gamma/(r*K);
            gr_new = zeros(d);
            id_client = datasample(1:m, m_per_round, 'Replace', false);
            for i = id_client
                gr_client = gr;
                wr_i = wr; %same starting point
                for k = 1:K
                    ar_i = gr_client/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); %parameter in client's optimization
                    wr_i = nuc_prox_cs(-ar_i/br, lambda_s/br, x0, rho); % primal update of client
        
                    G_i = nuc_grad(wr_i, bs, Data_set{i}); % compute gradients
                    gr_client = gr_client + G_i; % cumulative gradient of cilent
                end
                gr_new = gr_new + gr_client;
            end
            gr = gr_new/m_per_round; % aggregate gradients
            ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
            wr = nuc_prox_cs(-ar/br, lambda_s/br, x0, rho); % primal update of server
            r_now = r_now + 1;
            res_MFedDA(r_now, :) = eval_metric_nuc(wr, W_real, Data_set);
            if mod(r, 100) == 0
                disp(norm(wr - W_real, 'fro'))
            end
            xr = xr + wr;
        end
    end

    eval_list = cell(5, 1);
    eval_list{1} = res_FedMid;
    eval_list{2} = res_FedDA;
    eval_list{3} = res_FastFedDA;
    eval_list{4} = res_SFedDA;
    eval_list{5} = res_MFedDA;

    % save and plot results
    plot_nuclear(eval_list, plot_path);
    results = struct('eval_list', eval_list);
    save(results_path, 'results');
