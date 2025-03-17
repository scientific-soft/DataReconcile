% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для Dempster-Shafer (структура Демпстера-Шафера)
classdef DempsterShafer
    properties
        Intervals   % массив интервалов для фокальных элекментов [N×2]
        Masses      % соответствующие им массы [N×1]
    end
    
    properties (Constant)
        tolerance_threshold = 0.2;
    end

    methods
        % Конструктор
        function obj = DempsterShafer(intervals, masses)
            % Проверка входных аргументов
            if size(intervals, 2) ~= 2
                error('Поле Intervals должно быть матрицей размера N×2.');
            end
            if any(intervals(:,1) > intervals(:,2))
                error('В поле Interval левые границы интервалов должны быть меньше правых границ.');
            end
            if size(intervals, 1) ~= length(masses)
                error('Количество фокальныъ элементов должно совпадать с переданным числом масс.');
            end
            if any(masses < 0 | masses > 1)
                error('Значения в поле Masses должны быть от 0 до 1.');
            end
            threshold = 10^-2;
            if abs(sum(masses)-1) > threshold
                error('Сумма масс должна быть равна 1 (текущая сумма: %.6f)', sum(masses));
            end
            
            obj.Intervals = intervals;
            obj.Masses = masses(:);  % превращение в вектор-столбец.
        end
        
        % Сложение по правилу Демпстера (нормализированное)
        function Z = plus(A, B)
            if ~isa(A, 'DempsterShafer') || ~isa(B, 'DempsterShafer')
                error('Оба операнда должны быть объектами типа DempsterShafer.');
            end
            
            % Вычисление всех возможных сумм грани интервалов
            [n, m] = meshgrid(1:size(A.Intervals,1), 1:size(B.Intervals,1));
            n = n(:);
            m = m(:);
            
            % Вычисление комбинированных интервалов
            sum_intervals = A.Intervals(n,:) + B.Intervals(m,:);
            
            % Вычисление комбинированных масс (произведение)
            comb_masses = A.Masses(n) .* B.Masses(m);
            
            % Во избежание комбинаторного взрыва, объединяем близкие границы
            [unique_int, ~, ic] = uniquetol(sum_intervals, A.tolerance_threshold, 'ByRows', true);
            sum_masses = accumarray(ic, comb_masses);
            
            % Нормализация после агрегации
            if length(sum_masses) < length(comb_masses)
                sum_masses = sum_masses / sum(sum_masses);
            end
            
            Z = DempsterShafer(unique_int, sum_masses);
        end
        
        % Умножение на константу
        function Z = mtimes(k, A)
            if ~isa(A, 'Hist') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            z_intervals = k * A.Intervals;
            if k < 0
                z_intervals = z_intervals(:,[2,1]);
            end
            Z = DempsterShafer(z_intervals, A.Masses);
        end

        function stdCint = getStd(obj, method)
            intervals = obj.Intervals; % переменная для сокращенной записи
            masses = obj.Masses;       % переменная для сокращенной записи
            n = size(intervals, 1);
        
            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))

                stdCint = NaN(1,2);
                % 1. минимальное значение среднеквадратического отклонения
                % минимальная дисперсия внутриинтервального распределения (0) + дисперсия центров интервалов
                midpoints = mean(intervals, 2);
                mean_mid = masses' * midpoints;
                between_var = sum(masses .* (midpoints - mean_mid).^2);
                stdCint(1) = sqrt(between_var);
                
                % 2. максимальное значение дисперсии
                % максимальная дисперсия внутриинтервального распределения + дисперсия центров интервалов
                within_var = sum(masses .* diff(intervals,1,2).^2 / 4); 
                % случай, когда имеет место равномерное дисркетное распределение, сосредоточенное в границах фокального интервала
                stdCint(2) = sqrt(within_var + between_var);
        
            elseif strcmp(method,'accurate')
              
                % оптимизация
                % 1. минимальное значение дисперсии
                H = 2 * (diag(masses) - masses * masses');
                lb = intervals(:,1);  % левые границы интервалов
                ub = intervals(:,2);  % правые границы интервалов
                
                options = optimoptions('quadprog', 'Display', 'off');
                x_opt = quadprog(H, [], [], [], [], [], lb, ub, [], options);            
                mean_min = masses'*x_opt;
                minVar = sum(masses.*(x_opt-mean_min).^2);
                
                % оптимизация
                % 1. максимальное значение дисперсии
                maxVar = getStd(obj,'approximate'); 
                maxVar = maxVar(2)^2;   % начальное приближение

                % определение оптимальной точки (комбинаторный путь)
                if n <= 15  % практическое ограничение во избежание комбинаторного взрыва
                    % генерация всех возможных граничных точек (2^n)
                    combinations = dec2bin(0:2^n-1)-'0';
                    combinations = combinations(:,end:-1:1);  % двоичный порядок
                    
                    % конвертация в граничные точки
                    x_vals = repmat(intervals(:,1)',2^n,1)+repmat((intervals(:,2)-intervals(:,1))',2^n,1).*combinations;
                    
                    % вычисление дисперсии для всех комбинаций
                    for i = 1:size(combinations, 1)
                        x = x_vals(i,:)';
                        current_mean = masses'*x;
                        current_var = sum(masses.*(x-current_mean).^2);                    
                        if current_var > maxVar
                            maxVar = current_var;
                        end
                    end
                else
                    % Проверка экстремальных комбинаций граничных точек (ускорение)
                    x_left = intervals(:,1);
                    var_left = sum(masses.*(x_left-mean(x_left)).^2);
                    x_right = intervals(:,2);
                    var_right = sum(masses.*(x_right-mean(x_right)).^2);                
                    maxVar = max([maxVar, var_left, var_right]);
                end
    
                stdCint = sqrt([minVar, maxVar]);

            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end

        % Построение графика фокальных элементов с массами
        function plot(obj)
            figure;
            hold on;
            
            % Сортировка интервалов для лучшей визуализации
            [sorted_int, order] = sortrows(obj.Intervals);
            sorted_masses = obj.Masses(order);
            
            % Отображение каждого фокального элемента
            for i = 1:size(sorted_int,1)
                int = sorted_int(i,:);
                mass = sorted_masses(i);
                plot(int, [mass mass], 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerSize', 4);                
                % Добавление текста о массах
                text(mean(int), mass + 0.02, sprintf('%.3f', mass), 'HorizontalAlignment', 'center');
            end
            
            % Форматирование графика
            ylim([0 max(obj.Masses)*(1+obj.tolerance_threshold)]);
            xlabel('Interval');
            ylabel('Mass');
            title('Структура Демпстера-Шафера.');
            grid on;
            hold off;
        end

        % преобразование в PBox
        function pbox = DempsterShafer2PBox(obj, numPoints)
            if nargin < 2
                numPoints = length(obj.Masses);
            end
            % дискретизация носителя
            x_vals = linspace(min(obj.Intervals(:)), max(obj.Intervals(:)), numPoints);            
            borders = zeros(numPoints, 2);

            for i = 1:numPoints
                x = x_vals(i);
                % belief (нижняя граница CDF)
                borders(i,1) = sum(obj.Masses(obj.Intervals(:,1) <= x));
                % plausibility (верхняя граница CDF)
                borders(numPoints-i+1,2) = sum(obj.Masses(obj.Intervals(:,2) >= x));
            end
            lowerCDF = min(borders,[],2);
            upperCDF = max(borders,[],2);
            pbox = PBox(x_vals, lowerCDF, upperCDF);
        end

        % преобразование в Hist
        function Histogram = DempsterShafer2Hist(obj, numBins)
            if nargin < 2
               numBins = length(obj.Intervals);
            end
            % определение границ полос гистограммы
            all_points = [obj.Intervals(:,1); obj.Intervals(:,2)];
            edges = linspace(min(all_points), max(all_points), numBins+1);
            
            lowerPDF = zeros(1, numBins);
            upperPDF = zeros(1, numBins);
            
            for i = 1:numBins
                left = edges(i);
                right = edges(i+1);
                
                % оценка границ вероятностей
                lowerPDF(i) = sum(obj.Masses(obj.Intervals(:,1) >= left & obj.Intervals(:,2) <= right));
                upperPDF(i) = sum(obj.Masses(obj.Intervals(:,1) <= right & obj.Intervals(:,2) >= left));
            end
            
            % нормализация по длине полос
            bin_widths = diff(edges);
            lowerPDF = lowerPDF ./ bin_widths;
            upperPDF = upperPDF ./ bin_widths;
            
            Histogram = Hist(edges, lowerPDF, upperPDF);
        end

        % преобразование в Fuzzy
        function fv = DempsterShafer2Fuzzy(obj, numPoints)
            if nargin < 2
                numPoints = length(obj.Masses);
            end
            % дискретизация носителя
            x_vals = linspace(min(obj.Intervals(:)), max(obj.Intervals(:)), numPoints);
            membership = zeros(size(x_vals));
            
            % вычисление plausibility для каждой точки
            for i = 1:length(x_vals)
                x = x_vals(i);
                membership(i) = sum(obj.Masses(obj.Intervals(:,1) <= x & obj.Intervals(:,2) >= x));
            end
            
            fv = Fuzzy(x_vals, membership/max(membership));
        end

        % преобразование в FuzzyInterval
        function fi = DempsterShafer2FuzzyInterval(obj)
            % используем значения масс в качестве уровней значимости
            [sortedMasses, idx] = sort(obj.Masses, 'descend');
            alphaLevels = cumsum(sortedMasses);
            
            % вычисление вложенных интервалов
            intervals = obj.Intervals(idx,:);
            for i = 2:size(intervals,1)
                intervals(i,1) = min(intervals(i-1,1), intervals(i,1));
                intervals(i,2) = max(intervals(i-1,2), intervals(i,2));
            end
            
            fi = FuzzyInterval(alphaLevels(end:-1:1), intervals);
        end
    end
end