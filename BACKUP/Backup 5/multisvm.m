function [models] = multisvm(TrainingSet,GroupTrain)%,TestSet)
%Models a given training set with a corresponding group vector and 
%classifies a given test set using an SVM classifier according to a 
%one vs. all relation. 
%
%This code was written by Cody Neuburger cneuburg@fau.edu
%Florida Atlantic University, Florida USA
%This code was adapted and cleaned from Anand Mishra's multisvm function
%found at http://www.mathworks.com/matlabcentral/fileexchange/33170-multi-class-support-vector-machine/

u=unique(GroupTrain);
numClasses=length(u);
dim=60;
%display(u);
%display(numClasses);
%result = zeros(length(TestSet(:,1)),1);
%display(length(TestSet(:,1)));
%display(result);
%build models
for k=1:numClasses
    %Vectorized statement that binarizes Group
    %where 1 is the current class and 0 is all other classes
    G1vAll=(GroupTrain==u(k));
    %display(G1vAll);
    models(k) = svmtrain(TrainingSet,G1vAll);
end
keyboard
fprintf('Reading test images... \n');

%{
x=VideoReader('G:\STUDY\WhereAmI-master\test\test.mp4');
k=x.NumberOfFrames;
display(k);
%}
% Read a video frame and run the detector.
videoFileReader = vision.VideoFileReader('G:\STUDY\WhereAmI-master\test\test2.mp4');

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',[400 400 videoInfo.VideoSize+30]);

% Track the face over successive video frames 
% until the video is finished.
while ~isDone(videoFileReader)

    % Extract the next video frame
    videoFrame = step(videoFileReader);

        I = imresize(videoFrame,[512 NaN]);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    Iout = I;
    fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
    for a=1:size(bbox,1)
        fprintf('Face %d -> ', a);
        X = imcrop(I,bbox(a,:));
        TI = imresize(X,[dim dim]);
        TJ = rgb2gray(TI);
         gaborArray = gaborFilterBank(5,8,39,39);
        TV = transpose(gaborFeatures(TJ,gaborArray,4,4));
        TV = double(TV);
        %u=unique(Y);
        %numClasses=length(u);
    display(a);
        for k=1:numClasses
            if(svmclassify(models(k),TV)) 
                display(k);
                break;
            else
                display('else');
            end
        end
        result = k;

        pos = bbox(a,:);
        pos(2) = pos(2) - 20;
        pos(1) = pos(1) + pos(3)/8;
        if result == 8
            name = 'Other';
        elseif result == 1
            name = 'Shukla';
        elseif result == 2
            name = 'Rahul';
        elseif result == 3 
            name = 'solanki';
        elseif result == 4
            name = 'manish';
        elseif result == 5
            name = 'dhiman';
        elseif result == 6
            name = 'mahesh';
        else
            name = 'parihar';
        end
        fprintf('Person: %s. \n', name);
        Iout = insertShape(Iout,'rectangle',bbox(a,:),'Color','yellow');
    
    Iout = insertText(Iout,pos(1:2),name);
        %videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,name);
 step(videoPlayer, Iout);
        
    end
   
    
 
    % Insert a bounding box around the object being tracked
    %videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,name);

    % Display the annotated video frame 
    % using the video player object
   

end

% Release resources
release(videoFileReader);
release(videoPlayer);
%{
for i=1:10:k;
    keyboard
    b = read(x,i);
%s = strcat('G:\STUDY\WhereAmI-master\test\test',int2str(i), '.jpg');%what is use of this IMG? And how to call this IMG and dim
I = imresize(b,[512 NaN]);
faceDetector = vision.CascadeObjectDetector();
bbox = step(faceDetector, I);
Iout = I;
fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
for a=1:size(bbox,1)
    fprintf('Face %d -> ', a);
    X = imcrop(I,bbox(a,:));
    TI = imresize(X,[dim dim]);
    TJ = rgb2gray(TI);
     gaborArray = gaborFilterBank(5,8,39,39);
    TV = transpose(gaborFeatures(TJ,gaborArray,4,4));
    TV = double(TV);
    %u=unique(Y);
    %numClasses=length(u);
display(a);
    for k=1:numClasses
        if(svmclassify(models(k),TV)) 
            display(k);
            break;
        else
            display('else');
        end
    end
    result = k;


    %GROUP = svmclassify(SVMSTRUCT,TV);
    fprintf('Match found!  ');
    pos = bbox(a,:);
    pos(2) = pos(2) - 20;
    pos(1) = pos(1) + pos(3)/8;
    if result == 5
        name = 'Other';
    elseif result == 1
        name = 'Shukla';
    elseif result == 2
        name = 'Rahul';
    elseif result == 3 
        name = 'solanki';
    else
        name = 'manish';
    end
    fprintf('Person: %s. \n', name);
    Iout = insertShape(Iout,'rectangle',bbox(a,:),'Color','yellow');
    
    Iout = insertText(Iout,pos(1:2),name);
end
figure, imshow(Iout);
fprintf('\n\n');
end
%}
%classify test cases
%{
for j=1:size(TestSet,1)
    for k=1:numClasses
        if(svmclassify(models(k),TestSet(j,:))) 
            break;
        end
    end
    result(j) = k;
end
%}