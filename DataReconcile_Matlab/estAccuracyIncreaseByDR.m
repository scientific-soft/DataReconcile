function [accuracy_increase_ratio, variances_of_DR_result] = ...
    estAccuracyIncreaseByDR(dep_func, ChosenMode, vals_for_reconc, params, cov_matrix)
    % Функция предназначена для метрологической оценки потенциала повышения точности 
    % результатов совместных измерений за счет учета известных функциональных
    % взаимосвязей между измеряемыми величинами. 
    % Также функция возвращает линейное приближение к дисперсиям результатов
    % согласования (получаемых оценок).
    
    % Используемый алгоритм основан на вероятностном определении погрешностей 
    % (определение погрешности как случайного отклонения результата измерения 
    % от истинного значения искомой величины).
    % В качестве мер погрешности согласуемых измерений и (при необходимости) 
    % параметров уравнений взаимосвязей используется КОВАРИАЦИОННАЯ МАТРИЦА
    % вектора согласуемых величин.
    
    % Далее DR = Data Reconciliation
    
    % Функция возвращает: 
    % accuracy_increase_ratios - *вектор коэффицентов повышения точности за счет DR* 
    % accuracy_increase_ratios = *СКО до согласования* ./ *СКО после согласования*  
    % variances_of_DR_results - *вектор дисперсий результатов согласования
    % результатов измерений и параметров (если необходимо) уравнений взаимосвязей*
    
    % Функции передаются:
    % dep_func - указатель на функционал, описывающий взаимосвязи между
    % согласуемыми величинами
    % vals_for_reconc - вектор результатов измерений и параметров, которые
    % используются при согласовании через модель dep_func() = 0
    % cov_matrix - ковариационная матрица вектора согласуемых величин vals_to_be_reconc
    % Если одна из величин известна точно (предполагается, что одна из величин в 
    % уравнении/ях известна точно, либо ее значение не корректируется в рамках
    % процедуры согласования), тогда соотвествующую этой величине
    % дисперсию в диагонали матрицы нужно задать равной 0.
    % accuracy_increase_ratios такой величины будет равен 1.0,
    % то есть ее величина не коректируется, погрешность остается неизменной.
    
    num_equations = length(dep_func(vals_for_reconc, params));
    num_vals = length(vals_for_reconc);
    
    variances_of_DR_result = zeros([num_vals,1]);
    accuracy_increase_ratio = zeros([num_vals,1]);
    dF_dxj = zeros([num_equations,1]);
    Jacobian = zeros([num_equations,num_vals - 1]);
    
    for j = 1 : num_vals % j - номер согласуемой переменной, для которой рассчитывается 
                         % коэффициент повышения точности за счет учета
                         % функциональтных взаимосвязей между величинами
        nums_without_j_th = [1 : 1 : j - 1, j + 1 : 1 : num_vals];
        % формирование ковариационной матрицы вектора величин
        % без j-ой согласуемой величины
        tmp_mx = cov_matrix;
        tmp_mx(j,:) = [];
        tmp_mx(:,j) = [];
        cov_matrix_without_j_th = tmp_mx;
        for i = 1 : num_equations
            dF_dxj(i) = get_imag_numeric_derivative(dep_func, vals_for_reconc, params, j, i);
            for j_ = 1 : num_vals - 1
                Jacobian(i, j_) = get_imag_numeric_derivative(dep_func, vals_for_reconc, params, nums_without_j_th(j_), i);
            end
        end
        if strcmp('LS', ChosenMode) % if Least Squares method is chosen 
            linear_est_of_indirect_measurement_variance = ...
                abs(dF_dxj'*Jacobian)*cov_matrix_without_j_th*abs(Jacobian'*dF_dxj)/((dF_dxj'*dF_dxj)^2);
        elseif strcmp('WLS', ChosenMode) % if Weghted Least Squares method is chosen 
            var_of_delta_x_j = zeros([num_equations, 1]);
            weights_for_WLS = zeros([1, num_equations]);
            for j_ = 1 : num_equations
                sum_df_j_dz = Jacobian(j_,:).^2*diag(cov_matrix_without_j_th);
                % variance of error of i-th equasion' solution for x_j
                var_of_delta_x_j(j_) = dF_dxj(j_)^(-2)*sum_df_j_dz;
            end
            for j_ = 1 : num_equations
                tmp_w = var_of_delta_x_j(j_)./var_of_delta_x_j;
                tmp_w(isnan(tmp_w)) = 1;
                weights_for_WLS(j_) = 1/sum(tmp_w);
            end
            linear_est_of_indirect_measurement_variance = weights_for_WLS.^2*var_of_delta_x_j;
        else
            fprintf('Unknown mode for linear estimation of varience of system solution');
        end
    
        if linear_est_of_indirect_measurement_variance == 0
        fprintf('The true value of %d-th measurand can be derived from given dependensies equations. No reconciliation is needed for it''s measuarment result. \n', j);
        end
    
        accuracy_increase_ratio(j) = sqrt(1 + cov_matrix(j,j)/linear_est_of_indirect_measurement_variance);
        variances_of_DR_result(j) = cov_matrix(j,j)/accuracy_increase_ratio(j)^2;
    end
end

function df_dx = get_imag_numeric_derivative(func, params, params_, deriv_param_num, equation_num)
% func - указатель на функцию (в программном смысле) для одного из
% компонетов котрого нужно найти производную по deriv_param_num-му
% параметру нужно найти. Номер компонента задается переменной
% equation_num.
% params - вектор значений параметров/аргументов функций
% deriv_param_num - номер параметра, по которому нужно взять производную
% df_fx - численное значение производной функции
% equation_num - номер функции в системе уравнений F(X) = 0, для которйо
% нужно найти производную.
    alpha = params.*0;
    alpha(deriv_param_num) = params(deriv_param_num)*10^(-100) + 10^(-100); % конечно малое приращение в комплексной области
    df = func(params + alpha*1i, params_) - func(params, params_);
    df_dx = imag(df(equation_num))/alpha(deriv_param_num); % производная методом комплексного приращения
end











