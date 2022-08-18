function plot1 = plot_emnist(eval_list, plot_path, digits_only, eval_freq)
    res_FedMid = eval_list{1};
    res_FedDA = eval_list{2};
    res_FastFedDA = eval_list{3};
    res_SFedDA = eval_list{4};
    res_MFedDA = eval_list{5};

    if digits_only
        loss_lims = [0.38, 0.5];
        acc_lims = [0.84, 0.9];
    else
        loss_lims = [1.35, 1.6];
        acc_lims = [0.6, 0.66];
    end
    num_evals = size(res_FedMid, 1);
    x = 1:num_evals;
    x = x * eval_freq;

    emnist_plot = figure();
    emnist_plot.Position(3:4) = [800, 800];
    hold on;
    for i = 1:4
        subplot(2,2,i);
        YMatrix1 = [res_FedMid(:, i) res_FedDA(:, i) res_FastFedDA(:, i) res_SFedDA(:, i) res_MFedDA(:, i)];
        plot1 = plot(x, YMatrix1, 'LineWidth', 2);
        set(gca, 'LineWidth', 2);
        set(plot1(1),'DisplayName','FedMiD');
        set(plot1(2),'DisplayName','FedDA');
        set(plot1(3),'DisplayName','Fast-FedDA');
        set(plot1(4),'DisplayName','C-FedDA');
        set(plot1(5),'DisplayName','MC-FedDA');
        xlabel({'Round'}, 'FontSize', 12);
        if i == 1
            ylabel({'Train Loss'}, 'FontSize', 12);
            ylim(loss_lims);
        elseif i == 2
            ylabel({'Test Loss'}, 'FontSize', 12);
            ylim(loss_lims);
        elseif i == 3
            ylabel({'Train Accuracy'}, 'FontSize', 12);
            ylim(acc_lims);
        elseif i == 4
            ylabel({'Log Loss'}, 'FontSize', 12);
            ylim(acc_lims);
        end
        if i == 1
            legend('FontSize', 12);
        end
        xlim([0, num_evals * eval_freq]);
    end

    exportgraphics(emnist_plot, plot_path);
