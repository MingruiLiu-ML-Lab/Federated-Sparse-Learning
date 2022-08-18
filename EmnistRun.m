function eval_list = EmnistRun()

    rng('default');

    % hyperparameters and settings
    bs = 25; % batch size
    lambda = 1e-4; % regularization parameter
    R = 15000; % communication rounds
    m = 36; % clients per round
    K = 40; % updates per round
    eval_freq = 150;
    digits_only = true;
    small = false;
    plot_path = 'emnist_results.eps';
    results_path = 'emnist_results.mat';

    % data loading process
    if digits_only
        c = 10;
        if small
            M = 36; % number of clients
            n = 3884; % total number of training samples
            test_n = 436; % total number of testing samples
            data_path = 'data/FederatedEMNIST/small_emnist.mat';
        else
            M = 367; % number of clients
            n = 36259; % total number of training samples
            test_n = 4095; % total number of testing samples
            data_path = 'data/FederatedEMNIST/emnist.mat';
        end
    else
        c = 62;
        if small
            M = 36; % number of clients
            n = 5298; % total number of training samples
            test_n = 605; % total number of testing samples
            data_path = 'data/FederatedEMNIST62/small_emnist62.mat';
        else
            M = 379; % number of clients
            n = 73421; % total number of training samples
            test_n = 8346; % total number of testing samples
            data_path = 'data/FederatedEMNIST62/emnist62.mat';
        end
    end
    d = 784;
    X = zeros(d, n);
    Y = zeros(1, n);
    testX = zeros(d, test_n);
    testY = zeros(1, test_n);
    client_samples = zeros(1, M);
    test_client_samples = zeros(1, M);
    load(data_path);
    client_bounds = zeros(1,M+1);
    client_bounds(1) = 1;
    for i=2:M+1
        client_bounds(i) = client_bounds(i-1) + client_samples(i-1);
    end
    client_weights = double(client_samples) / sum(client_samples);

    num_methods = 5;
    if not(isfolder('results'))
        mkdir('results')
    end
    num_evals = R / eval_freq;
    eval_list = cell(num_methods, 1);
    current_method = 0;

    train_losses = zeros(num_methods, num_evals);
    test_losses = zeros(num_methods, num_evals);
    train_accs = zeros(num_methods, num_evals);
    test_accs = zeros(num_methods, num_evals);

    % FedMid
    disp('FedMid');
    current_method = current_method + 1;
    res_FedMid = zeros(num_evals, 4);

    etac = 0.01; % client learning rate
    etas = 1; % server learning rate
    wr = zeros(d, c, 1);
    W = zeros(d, c, m);
    Delta = zeros(d, c, 1);
    wtemp = zeros(d, c, 1);

    for r = 1:R
        round_clients = datasample(1:M, m, 'Replace', false);
        wtemp = wr; % record the snapshot
        W = repmat(wr, 1, 1, m);
        for i = 1:m
            client = round_clients(i);
            start_idx = client_bounds(client);
            end_idx = client_bounds(client+1) - 1;
            current_bs = bs;
            if current_bs > client_samples(client)
                current_bs = client_samples(client);
            end
            for k=1:K
                idx = datasample(start_idx:end_idx, current_bs, 'Replace', false);
                g = softmax_loss_grad(W(:,:,i), X(:,idx), Y(idx));
                W(:,:,i) = l1_soft(W(:,:,i) - etac*g, etac*lambda);
            end
        end
        round_weights = client_weights(round_clients);
        round_weights = round_weights / sum(round_weights);
        weighted_diff = reshape(round_weights, 1, 1, m).*(W - wtemp);
        Delta = sum(weighted_diff, 3);
        wr = l1_soft(wr + etas*Delta, etas*etac*lambda);
        if mod(r, eval_freq) == 0
            eval_idx = r / eval_freq;
            res_FedMid(eval_idx, :) = eval_metric_emnist(wr, X, Y, testX, testY);
            fprintf('round %d train loss, test loss, train acc, test acc: %.4f %.4f %.4f %.4f\n', r, res_FedMid(eval_idx, 1), res_FedMid(eval_idx, 2), res_FedMid(eval_idx, 3), res_FedMid(eval_idx, 4));
        end
    end
    eval_list{current_method} = res_FedMid;

    % FedDualAvg
    disp('FedDualAvg');
    current_method = current_method + 1;
    res_FedDA = zeros(num_evals, 4);

    etac = 0.01; % client learning rate
    etas = 1; % server learning rate
    z = zeros(d,c,1);
    Z = zeros(d,c,m);
    Delta = zeros(d,c,1);
    Ztemp = zeros(d,c,m);

    for r=1:R
        round_clients = datasample(1:M, m, 'Replace', false);
        Z=repmat(z, 1, 1, m);
        Ztemp = Z; % record the snapshot
        for i = 1:m
            client = round_clients(i);
            start_idx = client_bounds(client);
            end_idx = client_bounds(client+1) - 1;
            current_bs = bs;
            if current_bs > client_samples(client)
                current_bs = client_samples(client);
            end
            for k=1:K
                etark = etas * etac*r*K+etac*k;
                wtemp = l1_soft(Z(:,:,i),etark*lambda); % proximal mapping, retrieve primal
                idx = datasample(start_idx:end_idx, current_bs, 'Replace', false);
                g = softmax_loss_grad(wtemp,X(:,idx), Y(idx));
                Z(:,:,i) = Z(:,:,i)-etac*g; % client dual update
            end
        end
        round_weights = client_weights(round_clients);
        round_weights = round_weights / sum(round_weights);
        weighted_diff = reshape(round_weights, 1, 1, m).*(Z-Ztemp);
        Delta = sum(weighted_diff, 3); % correspond to delta_r
        z = z+etas*Delta; % server dual update;
        if mod(r, eval_freq) == 0
            eval_idx = r / eval_freq;
            wr = l1_soft(z, etas*etac*(r+1)*K*lambda);
            res_FedDA(eval_idx, :) = eval_metric_emnist(wr, X, Y, testX, testY);
            fprintf('round %d train loss, test loss, train acc, test acc: %.4f %.4f %.4f %.4f\n', r, res_FedDA(eval_idx, 1), res_FedDA(eval_idx, 2), res_FedDA(eval_idx, 3), res_FedDA(eval_idx, 4));
        end
    end
    eval_list{current_method} = res_FedDA;

    % Fast FedDualAvg with strong convexity (Our proposal)
    disp('Fast_FedDA');
    current_method = current_method + 1;
    res_FastFedDA = zeros(num_evals, 4);

    mu = 0.001;
    gamma = 25;
    x0 = zeros(d, c, 1);
    xr = zeros(d, c, 1); % cumulative primal variable
    wr = zeros(d, c, 1); % primal
    gr = zeros(d, c, 1); % cumulative gradient of server

    for r = 1:R
        round_clients = datasample(1:M, m, 'Replace', false);
        cr = lambda;
        gr_client = repmat(gr, 1, 1, m);
        xr_client = repmat(xr, 1, 1, m);
        for i = 1:m
            client = round_clients(i);
            wr_i = wr; %same starting point
            start_idx = client_bounds(client);
            end_idx = client_bounds(client+1) - 1;
            current_bs = bs;
            if current_bs > client_samples(client)
                current_bs = client_samples(client);
            end
            for k = 1:K
                idx = datasample(start_idx:end_idx, current_bs, 'Replace', false);
                G_i = softmax_loss_grad(wr_i, X(:,idx), Y(idx)); % compute gradients
                gr_client(:, :, i) = gr_client(:, :, i) + G_i; % cumulative gradient of cilent

                eta_k = (r-1)*K + k;
                ar_i = gr_client(:, :, i)/eta_k - 0.5*mu * xr_client(:, :, i)/eta_k - gamma * x0/eta_k; %parameter in client's optimization
                br_i = 0.5*mu + gamma/eta_k;
                wr_i = l1_soft(-ar_i/br_i, cr/br_i); % primal update of client
                xr_client(:, :, i) = xr_client(:, :, i) + wr_i;
            end
        end
        eta_r = r*K;
        round_weights = client_weights(round_clients);
        round_weights = round_weights / sum(round_weights);
        weighted_grads = reshape(round_weights, 1, 1, m).*gr_client;
        weighted_xr = reshape(round_weights, 1, 1, m).*xr_client;
        gr = sum(weighted_grads, 3); % aggregate gradients
        xr = sum(weighted_xr, 3);

        ar = gr/eta_r - 0.5*mu * xr/eta_r - gamma * x0/eta_r; % parameter in server's optimization
        br = 0.5*mu + gamma/eta_r;
        wr = l1_soft(-ar/br, cr/br); % primal update of client
        xr = xr + wr;
        if mod(r, eval_freq) == 0
            eval_idx = r / eval_freq;
            res_FastFedDA(eval_idx, :) = eval_metric_emnist(wr, X, Y, testX, testY);
            fprintf('round %d train loss, test loss, train acc, test acc: %.4f %.4f %.4f %.4f\n', r, res_FastFedDA(eval_idx, 1), res_FastFedDA(eval_idx, 2), res_FastFedDA(eval_idx, 3), res_FastFedDA(eval_idx, 4));
        end
    end
    eval_list{current_method} = res_FastFedDA;

    % FedDualAvg with strong convexity
    disp('SC_FedDA');
    current_method = current_method + 1;
    res_AFedDA = zeros(num_evals, 4);

    mu = 0.001;
    gamma = 25;
    x0 = zeros(d, c, 1);
    xr = zeros(d, c, 1); % cumulative primal
    wr = zeros(d, c, 1); % primal
    gr = zeros(d, c, 1); % gradient

    for r = 1:R
        round_clients = datasample(1:M, m, 'Replace', false);
        br = 0.5*mu + gamma/(r*K);
        cr = lambda;
        gr_client = repmat(gr, 1, 1, m);
        for i = 1:m
            wr_i = wr; %same starting point
            client = round_clients(i);
            start_idx = client_bounds(client);
            end_idx = client_bounds(client+1) - 1;
            current_bs = bs;
            if current_bs > client_samples(client)
                current_bs = client_samples(client);
            end
            for k = 1:K
                idx = datasample(start_idx:end_idx, current_bs, 'Replace', false);
                G_i = softmax_loss_grad(wr_i, X(:,idx), Y(idx)); % compute gradients
                gr_client(:, :, i) = gr_client(:, :, i) + G_i; % cumulative gradient of cilent

                ar_i = gr_client(:, :, i)/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); %parameter in client's optimization
                wr_i = l1_soft(-ar_i/br, cr/br); % primal update of client
            end
        end
        round_weights = client_weights(round_clients);
        round_weights = round_weights / sum(round_weights);
        weighted_grads = reshape(round_weights, 1, 1, m).*gr_client;
        gr = sum(weighted_grads, 3); % aggregate gradients
        ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
        wr = l1_soft(-ar/br, cr/br); % primal update of client
        xr = xr + wr;
        if mod(r, eval_freq) == 0
            eval_idx = r / eval_freq;
            res_AFedDA(eval_idx, :) = eval_metric_emnist(wr, X, Y, testX, testY);
            fprintf('round %d train loss, test loss, train acc, test acc: %.4f %.4f %.4f %.4f\n', r, res_AFedDA(eval_idx, 1), res_AFedDA(eval_idx, 2), res_AFedDA(eval_idx, 3), res_AFedDA(eval_idx, 4));
        end
    end
    eval_list{current_method} = res_AFedDA;

    % Multi-stage FedDualAvg with strong convexity
    disp('MC_FedDA');
    current_method = current_method + 1;
    res_MFedDA = zeros(num_evals, 4);

    mu = 0.001;
    gamma = 25;
    lambdas = [4e-4 2e-4 1e-4 1e-4 1e-4];
    x0 = zeros(d, c, 1);
    xr = zeros(d, c, 1); % cumulative primal variable
    wr = zeros(d, c, 1); % primal
    gr = zeros(d, c, 1); % cumulative gradient of server
    S = 5;
    R0 = R/S;

    for s = 1:S
        x0 = wr;
        xr = wr;
        gr = zeros(d, c, 1);
        rho = 1e5*(0.5)^s;
        lambda_s = lambdas(s);
        for r = 1:R0
            br = 0.5*mu + gamma/(r*K);
            cr = lambda_s;
            round_clients = datasample(1:M, m, 'Replace', false);
            gr_client = repmat(gr, 1, 1, m);
            for i = 1:m
                client = round_clients(i);
                wr_i = wr; %same starting point
                start_idx = client_bounds(client);
                end_idx = client_bounds(client+1) - 1;
                current_bs = bs;
                if current_bs > client_samples(client)
                    current_bs = client_samples(client);
                end
                for k = 1:K
                    idx = datasample(start_idx:end_idx, current_bs, 'Replace', false);
                    G_i = softmax_loss_grad(wr_i, X(:, idx), Y(idx));
                    gr_client(:, :, i) = gr_client(:, :, i) + G_i;

                    ar_i = gr_client(:, :, i)/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K);
                    wr_i = l1_soft_cs(-ar_i/br, cr/br, x0, rho);
                end
            end
            round_weights = client_weights(round_clients);
            round_weights = round_weights / sum(round_weights);
            weighted_grads = reshape(round_weights, 1, 1, m).*gr_client;
            gr = sum(weighted_grads, 3); % aggregate gradients
            ar = gr/(r*K) - 0.5*mu * xr/r - gamma * x0/(r*K); % parameter in server's optimization
            wr = l1_soft_cs(-ar/br, cr/br, x0, rho);
            xr = xr + wr;
            total_r = (s-1)*R0 + r;
            if mod(total_r, eval_freq) == 0
                eval_idx = total_r / eval_freq;
                res_MFedDA(eval_idx, :) = eval_metric_emnist(wr, X, Y, testX, testY);
                fprintf('round %d train loss, test loss, train acc, test acc: %.4f %.4f %.4f %.4f\n', total_r, res_MFedDA(eval_idx, 1), res_MFedDA(eval_idx, 2), res_MFedDA(eval_idx, 3), res_MFedDA(eval_idx, 4));
            end
        end
    end
    eval_list{current_method} = res_MFedDA;

    % save and plot results
    plot_emnist(eval_list, plot_path, digits_only, eval_freq);
    results = struct('eval_list', eval_list, 'digits_only', digits_only, 'eval_freq', eval_freq);
    save(results_path, 'results');
