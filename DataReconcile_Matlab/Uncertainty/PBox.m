% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для p-box (область возможных значений для функции распределения, cdf)
classdef PBox   % дискретизированный p-box
    properties
        x           % сетка дискретизации значений аргумента cdf
        lowerCDF    % нижние границы p-box (наименьшие возможные значения cdf)
        upperCDF    % верхние границы p-box (наибольшие возможные значения cdf)
    end
    
    methods
        % конструктор
        function obj = PBox(x, lowerCDF, upperCDF)
            % проверка размерностей входных данных
            if length(x) ~= length(lowerCDF) || length(x) ~= length(upperCDF)
                error('x, lowerCDF и upperCDF должны быть векторами одинакового размера.');
            end
            if any(lowerCDF > upperCDF)
                error('Нижние границы p-box не должны превышать верхние границы p-box для всех значений аргументов.');
            end
            if any(diff(lowerCDF) < 0) || any(diff(upperCDF) < 0)
                error('Функции распределения не могут убывать.');
            end
            if any(lowerCDF < 0) || any(upperCDF < 0)
                error('Значения функции распределения не могут быть меньше 0.');
            end
            if any(lowerCDF > 1) || any(upperCDF > 1)
                error('Значения функции распределения не могут быть больше 1.');
            end
            obj.x = x(:)';               % принудительный перевод в вектор-строку
            obj.lowerCDF = lowerCDF(:)'; % принудительный перевод в вектор-строку
            obj.upperCDF = upperCDF(:)'; % принудительный перевод в вектор-строку
        end
        
        % оператор сложения двух p-boxes (операнды: Z = A + B)
        function Z = plus(A, B)
            if ~isa(A, 'PBox') || ~isa(B, 'PBox')
                error('Оба операнда должны быть объектами типа PBox.');
            end
            % определение сетки значений аргумента x для результата операции
            min_z = min(A.x) + min(B.x);
            max_z = max(A.x) + max(B.x);
            z_values = linspace(min_z, max_z, max([length(A.x), length(B.x)]));
            
            lowerZ = zeros(size(z_values));
            upperZ = zeros(size(z_values));
            
            for i = 1:length(z_values)
                z = z_values(i);
                % для всех значений их A.x вычисляем разности y=z-x
                x_A = A.x;
                y_B = z-x_A;
                
                % оценка значения cdf для B в точках y_B
                F_B_lower = interp1(B.x, B.lowerCDF, y_B, 'linear', 0);
                F_B_lower = max(min(F_B_lower, 1), 0); % приведение к [0,1]
                F_B_upper = interp1(B.x, B.upperCDF, y_B, 'linear', 1);
                F_B_upper = max(min(F_B_upper, 1), 0); % приведение к [0,1]
                
                % вычисление нижней границы для Z в точке z
                temp_lower = A.lowerCDF + F_B_lower - 1;
                lowerZ(i) = max([temp_lower, 0], [], 'all');
                
                % вычисление верхней границы для Z в точке z
                temp_upper = A.upperCDF + F_B_upper;
                upperZ(i) = min([temp_upper, 1], [], 'all');
            end
            
            % проверка, что функция распределения не убывает
            lowerZ = cummax(lowerZ);
            upperZ = cummax(upperZ);
            
            % приведение к [0,1]
            lowerZ = max(lowerZ, 0);
            upperZ = min(upperZ, 1);
            
            Z = PBox(z_values, lowerZ, upperZ);
        end
        
        function Z = mtimes(k, A)
            if ~isa(A, 'PBox') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            z_values = k * A.x;
            if k >= 0
               lowerZ = A.lowerCDF;
               upperZ = A.upperCDF;
            else
               z_values = z_values(end:-1:1);
               lowerZ = 1-A.upperCDF(end:-1:1);
               upperZ = 1-A.lowerCDF(end:-1:1); 
            end
            Z = PBox(z_values, lowerZ, upperZ);
        end

        % построение графика p-box
        function plot(obj)
            figure; hold on;
            stairs(obj.x, obj.lowerCDF, 'b', 'LineWidth', 1.5); 
            stairs(obj.x, obj.upperCDF, 'b', 'LineWidth', 1.5);
            xlabel('x'); ylabel('CDF');
            legend('нижняя граница P-Box', 'верхняя граница P-Box', 'Location', 'southeast');
            title('Границы P-box');
            grid on;
        end

        % определение величины возможного среднеквадратического отклонения.
        function stdCint = getStd(obj, method)
            n = length(obj.x);

            % приближенный вариант: согласно теореме Крейновича-Тао.
            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))
                
                ind_left_lo  = min(find(obj.lowerCDF<0.1, 1, 'last')+1, n);
                if isempty(ind_left_lo)
                    ind_left_lo = 1;
                end
                ind_left_hi  = min(find(obj.upperCDF<0.1, 1, 'last')+1, n);
                if isempty(ind_left_hi)
                    ind_left_hi = 1;
                end
                ind_right_lo = min(find(obj.lowerCDF<0.9, 1, 'last')+1, n);
                if isempty(ind_right_lo)
                    ind_right_lo = 1;
                end
                ind_right_hi = min(find(obj.upperCDF<0.9, 1, 'last')+1, n);
                if isempty(ind_right_hi)
                    ind_right_hi = 1;
                end
    
                stdCint = NaN(1,2);
                stdCint(1) = max(0.25*(obj.x(ind_right_hi)-obj.x(ind_left_lo)), 0);
                stdCint(2) = 0.25*(obj.x(ind_right_lo)-obj.x(ind_left_hi));
            
            elseif strcmp(method,'accurate')
              
                % переменные для сокращения записей
                x = obj.x(:)';
                lowerCDF = obj.lowerCDF(:)';
                upperCDF = obj.upperCDF(:)';
                
                % дискретизация в интервалы
                nIntervals = length(x) - 1;
                p_lower = zeros(1, nIntervals);
                p_upper = zeros(1, nIntervals);
                
                % вычисление границ интервалов вероятности
                for i = 1:nIntervals
                    p_lower(i) = max(lowerCDF(i+1) - upperCDF(i), 0);
                    p_upper(i) = upperCDF(i+1) - lowerCDF(i);
                end
                
                % проверки согласованности
                if sum(p_lower) > 1 || sum(p_upper) < 1
                    error('Несогласованный p-box: не включает ни одного распределения.');
                end
                
                % вычисление середин интервал для оценки дисперсии
                x_mid = 0.5 * (x(1:end-1) + x(2:end));
                
                % оптимизация
                % 1. минимальное значение дисперсии
                A = [-eye(nIntervals); eye(nIntervals)];
                b = [-p_lower(:); p_upper(:)];
                Aeq = ones(1, nIntervals);
                beq = 1;
                options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 10^4);
                
                objFun = @(p) (p' * (x_mid.^2)' - (p' * x_mid')^2); % для минимизации
                p0 = p_lower'; % начальное приближение
                if sum(p_lower)~=0 
                    p0 = p0 / sum(p_lower);
                end
                
                p_opt_min = fmincon(objFun, p0, [], [], Aeq, beq, p_lower', p_upper', [], options);
                minVar = objFun(p_opt_min);
                
                % 2. максимальное значение дисперсии
                objFun = @(p) -(p' * (x_mid.^2)' - (p' * x_mid')^2); % максимизация через минимизацию
                p0 = p_upper' / sum(p_upper); % начальное приближение            
                p_opt_max = fmincon(objFun, p0, [], [], Aeq, beq, p_lower', p_upper', [], options);
                maxVar = -objFun(p_opt_max); % обращение негативизации
    
                stdCint = sqrt([minVar, maxVar]);
            
            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end  

        % преобразование в Hist
        function Histogram = PBox2Hist(obj, numBins)
            if nargin < 2
               numBins = length(obj.x);
            end
            % дискретизация носителя
            x_min = min(obj.x);
            x_max = max(obj.x);
            edges = linspace(x_min, x_max, numBins+1);
            
            lowerPDF = zeros(1, numBins);
            upperPDF = zeros(1, numBins);
            
            for i = 1:numBins
                left = edges(i);
                right = edges(i+1);
                
                % вычисление значений CDF в границах полос гистограммы
                idx_left = find(obj.x >= left, 1);
                idx_right = find(obj.x >= right, 1);
                
                % валидация
                idx_left = max(1, min(idx_left, length(obj.x)));
                idx_right = max(1, min(idx_right, length(obj.x)));
                
                % вычисления границ для вероятности
                p_lower = max(obj.lowerCDF(idx_right) - obj.upperCDF(idx_left), 0);
                p_upper = min(obj.upperCDF(idx_right) - obj.lowerCDF(idx_left), 1);
                
                % определение границ для PDF
                bin_width = right - left;
                lowerPDF(i) = p_lower / bin_width;
                upperPDF(i) = p_upper / bin_width;
            end
            
            Histogram = Hist(edges, lowerPDF, upperPDF);
        end

        % преобразование в DempsterShafer
        function ds = PBox2DempsterShafer(obj, numFocal)
            if nargin < 2
                numFocal = length(obj.x);
            end
            % дискретизация в фокальные элементы
            x_vals = linspace(min(obj.x), max(obj.x), numFocal+1);
            intervals = [x_vals(1:end-1)' x_vals(2:end)'];
            prob_low = obj.lowerCDF(round(linspace(1, length(obj.x), numFocal+1)));
            prob_high = obj.upperCDF(round(linspace(1, length(obj.x), numFocal+1)));
            masses = diff(0.5*(prob_low+prob_high));
            
            % нормализация масс
            masses = masses / sum(masses);
            ds = DempsterShafer(intervals, masses);
        end

        % преобразование в Fuzzy
        function fv = PBox2Fuzzy(obj, numPoints)
            if nargin < 2
               numPoints = length(obj.x);
            end
            % дискретизация носителя х
            x_vals = linspace(min(obj.x), max(obj.x), numPoints);
            membership = NaN(size(x_vals));

            % вычисление значений функции принадлежности как (1-CDF) (мера возможности)
            lowerCDF_interp = interp1(obj.x, obj.lowerCDF, x_vals, 'linear', 'extrap');
            upperCDF_interp = interp1(obj.x, obj.upperCDF, x_vals, 'linear', 'extrap');
            ind1 = upperCDF_interp > 1-lowerCDF_interp;
            ind2 = upperCDF_interp <= 1-lowerCDF_interp;
            membership(ind1) = 1-lowerCDF_interp(ind1);
            membership(ind2) = upperCDF_interp(ind2);
            
            % валидация
            membership = membership / max(membership);  % приведение к [0,1].            
            fv = Fuzzy(x_vals, membership);
        end

        % преобразование в FuzzyInterval
        function fi = PBox2FuzzyInterval(obj, numAlpha)
            if nargin < 2
               numAlpha = length(obj.x);
            end

            % определение уровней значимости
            alphaLevels = linspace(0, 1, numAlpha);
            alphaLevels = sort(alphaLevels, 'descend');            
            intervals = zeros(numAlpha, 2);
            
            n = length(obj.x);
            for i = 1:numAlpha
                alpha = alphaLevels(i);
                % отыскание интервала, такого, что Lower CDF <= (1 - alpha) <= Upper CDF
                lower_bound = interp1(obj.upperCDF+(0:n-1)*eps, obj.x, 1-alpha, 'linear', 'extrap');
                upper_bound = interp1(obj.lowerCDF+(0:n-1)*eps, obj.x, 1-alpha, 'linear', 'extrap');
                
                intervals(i,:) = [lower_bound, upper_bound];
            end
            
            % Валидация структуры вложенных интервалов
            for i = 2:numAlpha
                intervals(i,1) = min(intervals(i-1,1), intervals(i,1));
                intervals(i,2) = max(intervals(i-1,2), intervals(i,2));
            end
            
            fi = FuzzyInterval(alphaLevels, intervals);
        end
    end
end