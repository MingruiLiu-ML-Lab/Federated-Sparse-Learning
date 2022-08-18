function g = nuc_grad(w, bs, data)
    % w: d-by-d
    % bs: batch size
    % data: client data set, nm-by-2 cell
    g = zeros(size(w));
    nm = length(data);
    idx = datasample(1:nm, bs, 'Replace', false); % sample minibatch size 1

    for i = 1:bs
        X = data{idx(i), 1};
        y = data{idx(i), 2};
        g = g + X.*(trace(X'*w) - y);
    end
    g = g/bs;