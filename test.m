% function image = read_middleBurry(fileName)

fileName = 'C:\Enliang\MATLAB\patchBased\fountain_yilin\origImage\fountain.txt';

imageIdx = 0;
fid = fopen(fileName, 'r');
assert(fid ~= 0);
tline = fgets(fid); % read the number of image
tline = fgets(fid);
while ischar(tline)
    imageIdx = imageIdx + 1;      
    words=regexp(tline,' +','split');
    
    image(imageIdx).imageName = words{1};
    K = str2double( words(2:10)); K = reshape(K, 3,3); K = K'; 
    image(imageIdx).K = K;
    
    image(imageIdx).K(1,3) = image(imageIdx).K(1,3) - 300;
    image(imageIdx).K(2,3) = image(imageIdx).K(2,3) - 1200;
    
    R = str2double( words(11:19)); R = reshape(R, 3,3); R = R';
    image(imageIdx).R = R;    
    image(imageIdx).T = str2double( words(20:22))';
    image(imageIdx).C = -image(imageIdx).R' * image(imageIdx).T;
    
    image(imageIdx).imageData = imread(image(imageIdx).imageName);
    [hh, ww, dd] = size(image(imageIdx).imageData);
    image(imageIdx).h = hh; 
    image(imageIdx).w = ww; 
    image(imageIdx).d = dd;
    
    tline = fgets(fid);
end
fclose(fid);


depthMap = image(1).imageData;
a = patchMatch(image, depthMap);
figure(); imshow(a);

