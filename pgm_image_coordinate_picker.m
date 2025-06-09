function pgm_image_coordinate_picker()
    % 选择PGM图像文件
    [filename, pathname] = uigetfile('*.pgm', 'Select PGM Image File');
    if isequal(filename, 0)
        disp('User canceled file selection.');
        return;
    end
    filepath = fullfile(pathname, filename);
    
    try
        % 读取PGM图像
        img = imread(filepath);
        
        % 获取并显示图像分辨率
        [height, width] = size(img);
        disp(['Image resolution: ' num2str(width) ' x ' num2str(height)]);
        
        % 创建图形窗口
        fig = figure('Name', 'PGM Image Coordinate Picker', 'NumberTitle', 'off');
        imshow(img);
        title('Click two points on the image (press Enter after second click)');
        hold on;
        
        % 用户交互选取点
        disp('Select two points on the image...');
        [x, y] = ginput(2);
        
        % 验证输入
        if length(x) < 2
            close(fig);
            error('Only one point selected. Please select exactly two points.');
        end
        
        % 绘制点和连线
        plot(x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
        plot(x, y, 'g-', 'LineWidth', 1.5);
        
        % 显示坐标文本
        text(x(1), y(1)-15, sprintf('(%d, %d)', round(x(1)), round(y(1))), ...
            'Color', 'yellow', 'FontWeight', 'bold');
        text(x(2), y(2)-15, sprintf('(%d, %d)', round(x(2)), round(y(2))), ...
            'Color', 'yellow', 'FontWeight', 'bold');
        
        % 输出坐标结果
        point1 = [round(x(1)), round(y(1))];
        point2 = [round(x(2)), round(y(2))];
        
        disp('Selected coordinates:');
        disp(['Point 1: (x, y) = (' num2str(point1(1)) ', ' num2str(point1(2)) ')']);
        disp(['Point 2: (x, y) = (' num2str(point2(1)) ', ' num2str(point2(2)) ')']);
        
    catch ME
        disp(['Error: ' ME.message]);
    end
end