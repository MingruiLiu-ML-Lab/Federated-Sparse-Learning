function value = eval_metric_nuc(w, W_real, data)
    % w: current estimator
    % W_real: ground truth
    % data: training data

    Fro_norm = norm(w-W_real, 'fro'); % Frobenius norm
    L2_norm = norm(w-W_real); % operator norm
    S = svd(w);
    r = rank(w);
    loss = 0;

    m = length(data);
    n = 0;
    for i = 1:m
        client_data = data{i};
        nm = length(client_data);
        for j = 1:nm
            loss = loss + (client_data{j, 2} - trace(client_data{j, 1}'*w))^2;
        end
        n = n + nm;
    end
    value = [Fro_norm L2_norm r loss/n];


