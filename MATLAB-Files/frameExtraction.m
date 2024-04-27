% Specify your video file
videoFile = 'rawVideo.mp4';

% Create a VideoReader object
v = VideoReader(videoFile);

% Specify the folder to save the frames
outputFolder = '/Users/lukeludwig/Documents/MATLAB/Examples/R2024a/vision/MonocularVisualSimultaneousLocalizationAndMappingExample_myExample/ImageFolder';

% Frame counter
k = 0;

% Loop over the video frames
while hasFrame(v)
    % Read the next frame
    frame = readFrame(v);
    
    % Increment the frame counter
    k = k + 1;
    
    % Check if this is the 15th frame
    if mod(k, 15) == 0
        % Construct the output image file name
        outputImage = fullfile(outputFolder, sprintf('frame_%04d.jpg', k));
        
        % Write the frame to the output image file
        imwrite(frame, outputImage);
    end
end
