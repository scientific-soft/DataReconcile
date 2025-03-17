% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для переменных типа Fuzzy (нечеткие переменные)
classdef Fuzzy
    properties
        Universe      % значения носителя (вектор 1xN)
        Membership    % значения функции принадлежности (вектор 1xN, значения из [0,1])
    end
    
    methods
        % Конструктор
        function obj = Fuzzy(universe, membership)
            % Проверка входных переменных
            universe = universe(:)';        % перевод в вектор-строку
            membership = membership(:)';    % перевод в вектор-строку
            
            if length(universe) ~= length(membership)
                error('Поля Universe и Membership должны быть одинаковой длины.');
            end
            
            if any(membership < 0 | membership > 1)
                error('Значения функции принадлежности должны быть от 0 до 1.');
            end
            
            % порядок следования значений в поле universe - по возрастанию
            [obj.Universe, idx] = sort(universe);
            obj.Membership = membership(idx);
        end
        
        % Сложение с использованием принципа Заде (правило max-min)
        function Z = plus(A, B)
            % Создание результирующей переменной (поле Universe)
            min_z = A.Universe(1) + B.Universe(1);
            max_z = A.Universe(end) + B.Universe(end);
            z_step = min(diff(A.Universe(1:2)), diff(B.Universe(1:2)));
            z_universe = min_z:z_step:max_z;
            
            % Выделение памяти под функцию принадлежности
            z_membership = zeros(size(z_universe));
            
            % Принцип Заде (композиция)
            for i = 1:length(z_universe)
                z = z_universe(i);
                
                % Все возможные пары значений (x,y), для которых x+y=z.
                x_vals = A.Universe;
                y_vals = z-x_vals;
                
                % определение значений функции принадлежности для значений y_vals в B
                y_membership = interp1(B.Universe, B.Membership, y_vals, 'linear', 0);
                
                % вычисление минимакса от функции принадлежности 
                combined = min(A.Membership, y_membership);                
                z_membership(i) = max(combined);
            end
            
            Z = Fuzzy(z_universe, z_membership);
        end
        
        % Умножение на коэффициент
        function Z = mtimes(k, A)
            if ~isa(A, 'Fuzzy') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            z_membership = A.Membership;
            z_universe = k * A.Universe;
            if k < 0
                z_universe = z_universe(end:-1:1);
                z_membership = z_membership(end:-1:1);
            end
            Z = Fuzzy(z_universe, z_membership);
        end

        % вычисление вложенного (α-cut) интервала
        function interval = alphaCut(obj, alpha)
            % валидация значения α
            if alpha < 0 || alpha > 1
                error('Уровень значимости должен быть числом в интервале [0, 1].');
            end
            
            x = obj.Universe;       % переменная для краткости
            mu = obj.Membership;    % переменная для краткости
            
            % значения индексов, когда Membership >= alpha
            above = (mu >= alpha);            
            if ~any(above)
                interval = [NaN, NaN];
                return;
            end
            
            % левая граница
            first_idx = find(above, 1, 'first');
            if first_idx == 1
                left = x(1);
            else
                % линейная интерполяция
                x_prev = x(first_idx-1);
                x_curr = x(first_idx);
                mu_prev = mu(first_idx-1);
                mu_curr = mu(first_idx);
                left = x_prev+(alpha-mu_prev)*(x_curr-x_prev)/(mu_curr-mu_prev);
            end
            
            % правая граница
            last_idx = find(above, 1, 'last');
            if last_idx == length(x)
                right = x(end);
            else
                % линейная интерполяция
                x_curr = x(last_idx);
                x_next = x(last_idx+1);
                mu_curr = mu(last_idx);
                mu_next = mu(last_idx+1);                
                right = x_curr+(alpha-mu_curr)*(x_next-x_curr)/(mu_next-mu_curr);
            end
            
            interval = [left, right];
        end

        function stdCint = getStd(obj, method)
            
            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))
                % определение значений функции принадлежности для уровня α-cut, равного 0.05.
                int0 = alphaCut(obj, 0.05);
                int1 = alphaCut(obj, 1-eps);
                if any(isnan(int1))
                    stdCint = [0, 0.25*(int0(2)-int0(1))];
                else
                    val1 = 0.5 * (int1(1)-int0(1));
                    val2 = 0.5 * (int0(2)-int1(2));
                    stdCint = [0, mean([val1,val2])];
                end
            elseif strcmp(method,'accurate')
                % дискретизация уровней значимости
                alphaLevels = unique([0, obj.Membership, 1]);
                alphaLevels = sort(alphaLevels, 'descend');
                % дискретизация поля Universe
                x = obj.Universe;
                nBins = length(x)-1;
                binEdges = x;
                binMids = (x(1:end-1) + x(2:end)) / 2;
                
                % матрицы ограничений для оптимизации
                A = []; B = [];
                
                % построение ограничений из α-cuts
                for i = 1:length(alphaLevels)
                    alpha = alphaLevels(i);
                    interval = alphaCut(obj, alpha);
                    if any(isnan(interval))
                        continue;
                    end
                    a = interval(1);
                    b = interval(2);
                    inBin = (binEdges(1:end-1) >= a) & (binEdges(2:end) <= b);                    
                    % добавление ограничения: sum(p(inBin)) >= 1-alpha
                    A = [A; -double(inBin)];
                    B = [B; -(1 - alpha)];
                end
                
                % ограничение на нормировочное свойства 
                Aeq = ones(1, nBins);
                Beq = 1;
                
                % неотрицательность вероятностей
                lb = zeros(nBins, 1);
                ub = [];
                
                % оптимизация
                options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 10^4);

                % 1. минимальное значение дисперсии
                objFun = @(p) (p' * (binMids.^2)' - (p' * binMids')^2);
                p_min = fmincon(objFun, ones(nBins,1)/nBins, A, B, Aeq, Beq, lb, ub, [], options);
                minVar = objFun(p_min);
                if abs(minVar) < 10^-6  % устранение ошибок округления
                   minVar = 0;
                end

                % 2. максимальное значение дисперсии
                objFun = @(p) -(p' * (binMids.^2)' - (p' * binMids')^2);
                p_max = fmincon(objFun, ones(nBins,1)/nBins, A, B, Aeq, Beq, lb, ub, [], options);
                maxVar = -objFun(p_max);
            
                stdCint = sqrt([minVar, maxVar]);
            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end

        % Отображение функции принадлежности на графике
        function plot(obj)
            figure;
            area(obj.Universe, obj.Membership, 'FaceColor', [0.8 0.9 1], 'EdgeColor', 'b', 'LineWidth', 2);
            xlabel('Значения переменной');
            ylabel('Функция принадлежности');
            title('Нечеткая переменная');
            ylim([0 1+0.1]);
            grid on;
        end

        % преобразование в PBox
        function pbox = Fuzzy2PBox(obj, numPoints)
            if nargin < 2
                numPoints = length(obj.Universe);
            end
            x_vals = linspace(min(obj.Universe), max(obj.Universe), numPoints);
            lowerCDF = zeros(size(x_vals));
            upperCDF = zeros(size(x_vals));
            
            for i = 1:length(x_vals)
                x = x_vals(i);
                % нижняя граница CDF = 1 - Possibility(X > x)
                bound = 1 - max(obj.Membership(obj.Universe > x));
                if ~isempty(bound)
                   lowerCDF(i) = bound;
                else
                   lowerCDF(i) = 1;
                end
                % верхняя граница CDF = 1 - Necessity(X > x)
                bound = max(obj.Membership(obj.Universe <= x));
                if ~isempty(bound)
                    upperCDF(i) = bound;
                else 
                    upperCDF(i) = 0;
                end
            end            
            pbox = PBox(x_vals, lowerCDF, upperCDF);
        end

        % преобразование в Hist
        function Histogram = Fuzzy2Hist(obj, numBins)
            if nargin < 2
                numBins = length(obj.Universe);
            end
            edges = linspace(min(obj.Universe), max(obj.Universe), numBins+1);
            lowerProb = zeros(1, numBins);
            upperProb = zeros(1, numBins);
            
            for i = 1:numBins
                left = edges(i);
                right = edges(i+1);
                
                % определение полос
                in_bin = obj.Universe >= left & obj.Universe <= right;
                
                if any(in_bin)
                    lowerProb(i) = mean(in_bin) * min(obj.Membership(in_bin));
                    upperProb(i) = mean(in_bin) * max(obj.Membership(in_bin));
                end
            end
            
            % валидация
            if sum(lowerProb) > 1
                error('Оценка нижних границ вероятностей попадания в полосы содержит ошибку.');
            end

            % оценка значения PDF
            lowerPDF = lowerProb ./ diff(edges);
            upperPDF = upperProb ./ diff(edges);
            
            Histogram = Hist(edges, lowerPDF, upperPDF);
        end

        % преобразование в DempsterShafer
        function ds = Fuzzy2DempsterShafer(obj, numAlpha)
            if nargin < 2
                numAlpha = length(obj.Universe);
            end
            alphaLevels = linspace(0, 1, numAlpha+1);
            alphaLevels = alphaLevels(2:end);  % исключаем ноль.
            
            intervals = zeros(length(alphaLevels), 2);
            masses = diff([0 alphaLevels]);
            
            for i = 1:length(alphaLevels)
                interval = obj.alphaCut(alphaLevels(i));
                intervals(i,:) = interval;
            end
            
            ds = DempsterShafer(intervals, masses(:));
        end

        % преобразование в FuzzyInterval
        function fi = Fuzzy2FuzzyInterval(obj, numAlpha)
            if nargin < 2
                numAlpha = length(obj.Universe);
            end
            alphaLevels = linspace(0, 1, numAlpha);
            alphaLevels = sort(alphaLevels, 'descend');
            
            intervals = zeros(numAlpha, 2);
            for i = 1:numAlpha
                intervals(i,:) = obj.alphaCut(alphaLevels(i));
            end
            
            fi = FuzzyInterval(alphaLevels, intervals);
        end
    end
end