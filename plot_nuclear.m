function plot1 = plot_nuclear(eval_list, plot_path)
    res_FedMid = eval_list{1};
    res_FedDA = eval_list{2};
    res_FastFedDA = eval_list{3};
    res_SFedDA = eval_list{4};
    res_MFedDA = eval_list{5};
    nuc_plot = figure();
    nuc_plot.Position(3:4) = [800, 800];
    hold on;
    for i = 1:4
        YMatrix1 = [res_FedMid(:, i) res_FedDA(:, i) res_FastFedDA(:, i) res_SFedDA(:, i) res_MFedDA(:, i)];
        if i == 4
            YMatrix1 = log(YMatrix1);
        end
        subplot(2,2,i);
        plot1 = plot(YMatrix1,'LineWidth',2);
        set(plot1(1),'DisplayName','FedMiD');
        set(plot1(2),'DisplayName','FedDA');
        set(plot1(3),'DisplayName','Fast-FedDA');
        set(plot1(4),'DisplayName','C-FedDA');
        set(plot1(5),'DisplayName','MC-FedDA');
        xlabel({'Round'}, 'FontSize', 12);
        if i == 1
            ylabel({'Frobenius Error'}, 'FontSize', 12);
            ylim([0, 2]);
        elseif i == 2
            ylabel({'L2 Error'}, 'FontSize', 12);
            ylim([0, 0.6]);
        elseif i == 3
            ylabel({'Rank'}, 'FontSize', 12);
            ylim([15, 35]);
        elseif i == 4
            ylabel({'Log Loss'}, 'FontSize', 12);
            ylim([0, 1]);
        end
        
        if i == 1
            legend('FontSize', 12);
        end
    end

    exportgraphics(nuc_plot, plot_path);
