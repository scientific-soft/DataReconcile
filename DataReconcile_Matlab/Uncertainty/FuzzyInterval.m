% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для переменных типа FuzzyInterval (нечеткий интервал)
classdef FuzzyInterval
    properties
        AlphaLevels  % уровни α-cut (вектор 1×N)
        Intervals    % соответствующие вложенные интервалы (матрица N×2)
    end
    
    methods
        function obj = FuzzyInterval(alphaLevels, intervals)
            % Проверка входных аргументов
            if size(intervals,2) ~= 2
                error('Границы вложенных интервалов должны задаваться строкой из двух значений.');
            end
            if length(alphaLevels) ~= size(intervals, 1)
                error('Количество уровней alpha-cut должно соответствовать числу вложенных интервалов.');
            end
            if any(alphaLevels < 0 | alphaLevels > 1)
                error('Уровни alpha-cut должны иметь значения в пределах от 0 до 1.');
            end
            if any(intervals(:,1) > intervals(:,2))
                error('Правые границы вложенных интервалов должны быть меньше левых границ.');
            end
            
            % Сортировка по значения alpha-cut
            [obj.AlphaLevels, idx] = sort(alphaLevels(:)', 'descend');
            obj.Intervals = intervals(idx,:);

            % Проверка вложенного характера интервалов Intervals:
            % с уменьшением α-cut вложенные интервалы должны расширяться.
            if any(obj.Intervals(:,1)-sort(obj.Intervals(:,1),'descend')) || ...
               any(obj.Intervals(:,2)-sort(obj.Intervals(:,2),'ascend'))     
                error('Вложенность интервалов в поле Intervals нарушена.');
            end
        end
        
        % определение величины возможного среднеквадратического отклонения (согласно теореме Крейновича-Тао).
        function stdCint = getStd(obj, method)
            int = obj.Intervals;    % переменная для краткости
            a = obj.AlphaLevels;    % переменная для краткости
            widths = int(:,2)-int(:,1);     % ширина интервалов

            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))
                alpha = 0.05;
                ind1 = find(a == alpha, 1, 'first');
                ind2 = find(a == 1, 1, 'first');
                if ~isempty(ind1) && ~isempty(ind2)
                   stdCint = [0, 0.25*(widths(ind1)-widths(ind2))];
                else
                   ind1 = find(a>alpha, 1, 'last');
                   if ind1 == 1
                      stdCint = [0, 0.25*widths(ind1)];
                   else
                      % линейная интерполяция
                      width = (alpha-a(ind1-1)) * (widths(ind1)-widths(ind1-1)) / (a(ind1)-a(ind1-1)) + widths(ind1-1);
                      stdCint = [0, 0.25*width];
                   end
                   if ~isempty(ind2)
                      stdCint(2) = stdCint(2) - 0.25*widths(ind2);
                   end
                end
            
            elseif strcmp(method,'accurate')
                % дискретизация носителя, используя границы вложенных интервалов
                breakpoints = unique([obj.Intervals(:,1); obj.Intervals(:,2)]);
                nBins = length(breakpoints) - 1;
                binEdges = sort(breakpoints);
                
                % определение ограничений на вероятности из α-cuts
                A = [];
                b = [];
                for i = 1:length(obj.AlphaLevels)
                    alpha = obj.AlphaLevels(i);
                    a = obj.Intervals(i,1);
                    b_int = obj.Intervals(i,2);
                    inBin = (binEdges(1:end-1) >= a) & (binEdges(2:end) <= b_int);
                    
                    % добавление ограничения: sum(p(inBin)) >= 1 - alpha
                    A = [A; -double(inBin')];  % Преобразование к виду sum(p)>=1-alpha
                    b = [b; -(1-alpha)];
                end
                
                % добавление ограничений на сумму вероятностей
                Aeq = ones(1, nBins);
                beq = 1;
                % добавление ограничения на неотрицательность вероятностей
                lb = zeros(nBins, 1);
                ub = [];
                
                % параметры промежутков
                binMids = (binEdges(1:end-1) + binEdges(2:end))/2;
   
                options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 10^4);
    
                % 1. минимальное значение дисперсии
                objFun = @(p) (p'*(binMids.^2) - (p'*binMids)^2);
                p_min = fmincon(objFun, ones(nBins,1)/nBins, A, b, Aeq, beq, lb, ub, [], options);
                minVar = objFun(p_min);
                % 2. максимальное значение дисперсии
                objFun = @(p) -(p'*(binMids.^2) + (p'*binMids)^2);
                p_max = fmincon(objFun, ones(nBins,1)/nBins, A, b, Aeq, beq, lb, ub, [], options);
                maxVar = -objFun(p_max);
    
                stdCint = sqrt([minVar, maxVar]);
            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end
        
        % Сложение с использованием принципа Заде (правило max-min)
        function Z = plus(A, B)
            if ~isa(A, 'FuzzyInterval') || ~isa(B, 'FuzzyInterval')
                error('Оба операнда должны быть объектами типа FuzzyInterval.');
            end
            
            % комбинирование и сортировка уровней значимости
            combinedAlpha = unique([A.AlphaLevels, B.AlphaLevels]);
            combinedAlpha = sort(combinedAlpha, 'descend');
            
            % вычисление интервалов на каждом уровне значимости
            sumIntervals = zeros(length(combinedAlpha), 2);
            for i = 1:length(combinedAlpha)
                alpha = combinedAlpha(i);
                intA = A.getIntervalAtAlpha(alpha);
                intB = B.getIntervalAtAlpha(alpha);
                % суммирование интервалов по интервальной арифметике
                sumIntervals(i,:) = [intA(1)+intB(1), intA(2)+intB(2)];
            end
            
            Z = FuzzyInterval(combinedAlpha, sumIntervals);
        end
        
        % вложенный интервал на заданном уровне значимости (с интерполяцией)
        function interval = getIntervalAtAlpha(obj, alpha)
            if alpha < 0 || alpha > 1
                error('Значение alpha должно быть от 0 до 1.');
            end
            
            % граничные значения α
            idxHigh = find(obj.AlphaLevels >= alpha, 1, 'last');
            idxLow = find(obj.AlphaLevels <= alpha, 1, 'first');
            
            if isempty(idxHigh)
                interval = obj.Intervals(1,:);
            elseif isempty(idxLow) 
                interval = obj.Intervals(end,:);
            elseif idxHigh == idxLow  % точное соответствие
                interval = obj.Intervals(idxHigh,:);
            else  % линейная интерполяция
                alphaHigh = obj.AlphaLevels(idxHigh);
                alphaLow = obj.AlphaLevels(idxLow);
                weight = (alpha-alphaLow)/(alphaHigh-alphaLow);
                
                % интерполяция левых границ
                leftHigh = obj.Intervals(idxHigh,1);
                leftLow = obj.Intervals(idxLow,1);
                left = leftHigh + (leftLow-leftHigh)*(1-weight);
                
                % интерполяция правых границ
                rightHigh = obj.Intervals(idxHigh,2);
                rightLow = obj.Intervals(idxLow,2);
                right = rightHigh + (rightLow-rightHigh)*weight;
                
                interval = [left, right];
            end
        end

        % Умножение на коэффициент
        function Z = mtimes(k, A)
            if ~isa(A, 'FuzzyInterval') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            z_alphalevels = A.AlphaLevels;
            z_intervals = k * A.Intervals;
            if k < 0
                z_intervals = z_intervals(:,[2,1]);
            end
            Z = FuzzyInterval(z_alphalevels, z_intervals);
        end

        % Графическое отображение функции принадлежности нечеткого интервала
        function plot(obj)
            figure;
            hold on;
            
            % Отображение α-cuts
            for i = 1:length(obj.AlphaLevels)
                a = obj.Intervals(i,1);
                b = obj.Intervals(i,2);
                alpha = obj.AlphaLevels(i);
                plot([a b], [alpha alpha], 'b-', 'LineWidth', 1.5);
                
                % Отображение вертикальных соединительных линий
                if i < length(obj.AlphaLevels)
                    next_a = obj.Intervals(i+1,1);
                    next_b = obj.Intervals(i+1,2);
                    plot([a next_a], [alpha obj.AlphaLevels(i+1)], 'b--');
                    plot([b next_b], [alpha obj.AlphaLevels(i+1)], 'b--');
                end
            end
            
            xlabel('Значение');
            ylabel('\alpha');
            title('Нечеткий интервал.');
            grid on;
            ylim([0 1+0.1]);
        end

        % преобразование в PBox
        function pbox = FuzzyInterval2PBox(obj, numPoints)
            if nargin < 2
                numPoints = length(obj.AlphaLevels);
            end

            x_vals = linspace(min(obj.Intervals(:)), max(obj.Intervals(:)), numPoints);
            lowerCDF = zeros(size(x_vals));
            upperCDF = zeros(size(x_vals));
            
            for i = 1:length(x_vals)
                x = x_vals(i);
                bound = max(obj.AlphaLevels(obj.Intervals(:,1) < x));
                if ~isempty(bound)
                   upperCDF(i) = min(bound, 1);
                else
                   upperCDF(i) = 0;
                end
                bound = 1 - max(obj.AlphaLevels(obj.Intervals(:,2) > x));
                if ~isempty(bound)
                    lowerCDF(i) = max(bound, 0);
                else 
                    lowerCDF(i) = 1;
                end
            end
            
            pbox = PBox(x_vals, lowerCDF, upperCDF);
        end

        % преобразование в Hist
        function Histogram = FuzzyInterval2Hist(obj, numBins)
            if nargin < 2
                numBins = length(obj.AlphaLevels);
            end
            % определение границ полос гистограммы
            edges = linspace(min(obj.Intervals(:)), max(obj.Intervals(:)), numBins+1);
            
            lowerPDF = zeros(1, numBins);
            upperPDF = zeros(1, numBins);
            
            for i = 1:numBins
                left = edges(i);
                right = edges(i+1);
                
                % определение максимального уровня значимости, при котором интервал содержит данную полосу
                containing = (obj.Intervals(:,1) <= left) & (obj.Intervals(:,2) >= right);
                if any(containing)
                    max_alpha = max(obj.AlphaLevels(containing));
                else
                    max_alpha = 0;
                end
                
                % границы PDF пропорционально уровню значимости
                lowerPDF(i) = 0;  % консервативная нижняя граница
                upperPDF(i) = max_alpha;
            end
            
            % определение границ значений PDF
            bin_widths = diff(edges);
            total_upper = sum(upperPDF .* bin_widths);
            upperPDF = upperPDF / total_upper;
            
            Histogram = Hist(edges, lowerPDF, upperPDF);
        end

        % преобразование в DempsterShafer
        function ds = FuzzyInterval2DempsterShafer(obj, method)
            if (nargin < 2) || strcmp(method, 'fractional')
                % вычисление масс из разностей значений уровня значимости
                masses = -0.5*diff([obj.AlphaLevels']);            
                masses = [masses; masses];
                intervals = [];
                if obj.AlphaLevels(1) == 1
                   masses = [1.0; masses];
                   intervals = [obj.Intervals(1,:); intervals];
                end
                intervals = [intervals; [obj.Intervals(2:end,1), obj.Intervals(1:end-1,1)]];
                intervals = [intervals; [obj.Intervals(1:end-1,2), obj.Intervals(2:end,2)]];
                % удаление интервалов с нулевой массой
                valid = masses > 0;
                ds = DempsterShafer(intervals(valid,:), masses(valid)'/sum(masses));
            elseif strcmp(method, 'nested')
                masses = -diff(obj.AlphaLevels');
                % удаление интервалов с нулевой массой
                valid = masses > 0;
                ds = DempsterShafer(obj.Intervals(valid,:), masses(valid)'/sum(masses(valid)));
            else
                error('Поле method может быть равно только fractional или nested.');
            end
        end

        % преобразование в Fuzzy
        function fv = FuzzyInterval2Fuzzy(obj, numPoints)
            if nargin < 2
                numPoints = length(obj.AlphaLevels);
            end
            % дискретизация носителя
            x_vals = linspace(min(obj.Intervals(:)), max(obj.Intervals(:)), numPoints);
            membership = zeros(size(x_vals));
            
            % оценка уровня значимости как максимального alpha, при котором вложенный интервал содержит x
            for i = 1:length(x_vals)
                x = x_vals(i);
                contains_x = (obj.Intervals(:,1) <= x) & (obj.Intervals(:,2) >= x);
                if any(contains_x)
                    membership(i) = max(obj.AlphaLevels(contains_x));
                end
            end
            
            fv = Fuzzy(x_vals, membership);
        end
    end
end