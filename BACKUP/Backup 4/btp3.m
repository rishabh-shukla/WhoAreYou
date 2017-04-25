
clc;
%% Read Main subject frames from video.
fprintf('Initializing training on our First class.\n');

fprintf('Extracting features...  ');
tic;
n1=60;%set n1 value
dim=60;%set Dimension
temp=0;
for a=1:n1
    s = strcat('G:\STUDY\WhereAmI-master\positive\', int2str(a), '.jpg');
    %I = imresize(imread(s),[512 NaN]);
    I=imread(s);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    %fprintf('Total %d faces found. Beginning comparison..  \n',size(bbox,1));
    if size(bbox,1)==1
        for b=1:size(bbox,1);
        fprintf('%d ', a);
        X = imcrop(I,bbox(b,:));
        temp=temp+1;
        I1 = imresize(X,[dim dim]);
        J = rgb2gray(I1);
        gaborArray = gaborFilterBank(5,8,39,39);
        %test=transpose(gaborFeatures(J,gaborArray,4,4));
        %T(temp,:)=transpose(gaborFeatures(J,gaborArray,4,4));
        %test=transpose(hog_feature_vector(J));
        T(temp,:) = transpose(gaborFeatures(J,gaborArray,4,4)); %Reduce time this way. But how!??
        end
    end
    
end

fprintf('\nn1 is %d\n',temp);
toc;
n1=temp;
%% Read Negative Class. Covering all faces.
fprintf('Initializing training on our Second Class.\n');
fprintf('Extracting features...  ');
tic;
n2=60;
temp=n1;
for a=1:n2;
    s = strcat('G:\STUDY\WhereAmI-master\chunnu\', int2str(a),'.jpg');
    %I = imresize(imread(s),[512 NaN]);
    I=imread(s);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    %fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
    if size(bbox,1)==1
        for b=1:size(bbox,1);
        fprintf('%d ',a);
        X = imcrop(I,bbox(b,:));
        temp=temp+1;
        I1 = imresize(X,[dim dim]);
        J = rgb2gray(I1);
         gaborArray = gaborFilterBank(5,8,39,39);
        T(temp,:) = transpose(gaborFeatures(J,gaborArray,4,4)); %Reduce time this way. But how!
        end
    end
end
n2=temp-n1;
fprintf('\nn2 is %d\n',n2);
toc;
%% Read Third Class
fprintf('Initializing training on our Third Class.\n');
fprintf('Extracting features...  ');
tic;
n3=60;
temp=n1+n2;
for a=1:n3;
    s = strcat('G:\STUDY\WhereAmI-master\solanki\', int2str(a), '.jpg');
    %I = imresize(imread(s),[512 NaN]);
    I=imread(s);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    %fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
    if size(bbox,1)==1
        for b=1:size(bbox,1);
        fprintf('%d ',a);
        X = imcrop(I,bbox(b,:));
        temp=temp+1;
        I1 = imresize(X,[dim dim]);
        J = rgb2gray(I1);
         gaborArray = gaborFilterBank(5,8,39,39);
        T(temp,:) = transpose(gaborFeatures(J,gaborArray,4,4)); %Reduce time this way. But how!
        end
    end
end
n3=temp-n1-n2;
fprintf('\nn3 is %d\n',n3);
toc;
%% Read fourth class
fprintf('Initializing training on our Fourth Class.\n');
fprintf('Extracting features...  ');
tic;
n4=60;
temp=n1+n2+n3;
for a=1:n4;
    s = strcat('G:\STUDY\WhereAmI-master\manish\', int2str(a),'.jpg');
    %I = imresize(imread(s),[512 NaN]);
    I=imread(s);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    %fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
    if size(bbox,1)==1
        for b=1:size(bbox,1);
        fprintf('%d ',a);
        X = imcrop(I,bbox(b,:));
        temp=temp+1;
        I1 = imresize(X,[dim dim]);
        J = rgb2gray(I1);
         gaborArray = gaborFilterBank(5,8,39,39);
        T(temp,:) = transpose(gaborFeatures(J,gaborArray,4,4)); %Reduce time this way. But how!
        end
    end
end
n4=temp-n1-n2-n3;
fprintf('\nn4 is %d\n',n4);
toc;
%% Read fifth class
fprintf('Initializing training on our Fifth Class.\n');
fprintf('Extracting features...  ');
tic;
n5=60;
temp=n1+n2+n3+n4;
for a=1:n5;
    s = strcat('G:\STUDY\WhereAmI-master\negative\ne (', int2str(a),').jpg');
    %I = imresize(imread(s),[512 NaN]);
    I=imread(s);
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, I);
    %fprintf('Total %d faces found. Beginning comparison..  \n', size(bbox,1));
    if size(bbox,1)==1
        for b=1:size(bbox,1);
        fprintf('%d ',a);
        X = imcrop(I,bbox(b,:));
        temp=temp+1;
        I1 = imresize(X,[dim dim]);
        J = rgb2gray(I1);
         gaborArray = gaborFilterBank(5,8,39,39);
        T(temp,:) = transpose(gaborFeatures(J,gaborArray,4,4)); %Reduce time this way. But how!
        end
    end
end
n5=temp-n1-n2-n3-n4;
fprintf('\nn5 is %d\n',n5);
toc;
%% Train SVM

fprintf('%d %d %d %d %d\n',n1,n2,n3,n4,n5);
fprintf('Training:   ');
tic;
T = double(T);

Y = ones(n1+n2+n3+n4+n5,1); 
Y(n1+1:n1+n2) = ones(n2,1)*2;
Y(n1+n2+1:n1+n2+n3) = ones(n3,1)*3;
Y(n1+n2+n3+1:n1+n2+n3+n4) = ones(n4,1)*4;
Y(n1+n2+n3+n4+1:n1+n2+n3+n4+n5) = ones(n5,1)*5;
display(size(T));
display(size(Y));
%SVMSTRUCT = svmtrain(T,Y);
SVMSTRUCT=multisvm(T,Y);
toc;
fprintf('Training complete.\n\n');

    
%% Read Test Images
%{
fprintf('Reading test images... \n');
s = strcat('G:\STUDY\WhereAmI-master\test\test1', '.jpg');%what is use of this IMG? And how to call this IMG and dim
I = imresize(imread(s),[512 NaN]);
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
    u=unique(Y);
numClasses=length(u);

    
for j=1:size(TV,1)
    for k=1:numClasses
        if(svmclassify(models(k),TV(j,:))) 
            break;
        end
    end
    result(j) = k;
end

    %GROUP = svmclassify(SVMSTRUCT,TV);
    fprintf('Match found!  ');
    pos = bbox(a,:);
    pos(2) = pos(2) - 20;
    pos(1) = pos(1) + pos(3)/8;
    if GROUP == 0
        name = 'Other';
    else
        name = 'Shukla';
    end
    fprintf('Person: %s. \n', name);
    Iout = insertShape(Iout,'rectangle',bbox(a,:),'Color','yellow');
    Iout = insertText(Iout,pos(1:2),name);
end
figure, imshow(Iout);
fprintf('\n\n');

%}
