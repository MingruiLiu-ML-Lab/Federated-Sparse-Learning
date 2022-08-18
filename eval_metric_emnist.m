function value = eval_metric_emnist(w, X, Y, testX, testY)
    train_loss = softmax_loss(w, X, Y);
    test_loss = softmax_loss(w, testX, testY);
    train_acc = emnist_accuracy(w, X, Y);
    test_acc = emnist_accuracy(w, testX, testY);
    value = [train_loss test_loss train_acc test_acc];
