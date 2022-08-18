function plot1 = plot_lasso(eval_list, plot_path)
    res_FedMid = eval_list{1};
    res_FedDA = eval_list{2};
    res_FastFedDA = eval_list{3};
    res_SFedDA = eval_list{4};
    res_MFedDA = eval_list{5};
    lasso_plot = figure();
    lasso_plot.Position(3:4) = [800, 800];
    hold on;
    for i = 1:4
        subplot(2,2,i);
        YMatrix1 = [res_FedMid(:, i) res_FedDA(:, i) res_FastFedDA(:, i) res_SFedDA(:, i) res_MFedDA(:, i)];
        if i == 4
            YMatrix1 = log(YMatrix1);
        end
        plot1 = plot(YMatrix1, 'LineWidth', 2);
        set(plot1(1),'DisplayName','FedMiD');
        set(plot1(2),'DisplayName','FedDA');
        set(plot1(3),'DisplayName','Fast-FedDA');
        set(plot1(4),'DisplayName','C-FedDA');
        set(plot1(5),'DisplayName','MC-FedDA');
        xlabel({'Round'}, 'FontSize', 12);
        if i == 1
            ylabel({'L2 Error'}, 'FontSize', 12);
            ylim([0, 1.5]);
        elseif i == 2
            ylabel({'L1 Error'}, 'FontSize', 12);
            ylim([0, 30]);
        elseif i == 3
            ylabel({'F1 Score'}, 'FontSize', 12);
            ylim([0.65, 1.01]);
        elseif i == 4
            ylabel({'Log Loss'}, 'FontSize', 12);
            ylim([0, 1.5]);
        end
        if i == 1
            legend('FontSize', 12);
        end
    end

    exportgraphics(lasso_plot, plot_path);
