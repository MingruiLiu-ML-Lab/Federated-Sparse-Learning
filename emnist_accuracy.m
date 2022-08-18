function acc = emnist_accuracy(w,x,y)
    % w: (d, c)
    % x: (d, b)
    % y: (1, b)

    logits = x'*w;
    [max_logits, preds] = max(logits, [], 2);
    preds = reshape(preds, 1, size(y, 2));
    correct = (preds == (y + 1));
    acc = double(sum(correct)) / size(correct, 2);
end
