clear;


N = 40*5; %Number of training images. Same set as used in part 1
Ntest = 40*5; %Number of test images
d = 112*92;
A = zeros(d,N); %Store all the training images column wise in A
id = zeros(N,1); %Record the identity of each person
count = 0;

%Load the training images (First five images for each of 40 persons)
for i=1:40
    for j=1:5
        fname = sprintf ('att_faces/s%d/%d.pgm',i,j);
        im = double(imread(fname));   
        count = count+1;
        A(:,count) = im(:);
        id(count) = i;
        %In following line, record the identity of the person in array id
        
    end
end

%Perform the same steps as in part 1 to compute the eigencoefficients for
%the training images. These will be several lines of code taken from part 1.
meanA = mean(A, 2);
A = A - repmat(meanA, 1, N);
L = A.' * A;
[V,D] = eig(L);
V = A * V;
for i=1:N 
    V(:,i) = V(:,i)/norm(V(:,i)); 
end
V = V(:,end:-1:1);
D = diag(D); 
D = D(end:-1:1);
eigcoeffs_training = V.' * A; % This will be a matrix of size N x N 

%Test recognition rate for different values of k
for k = [1 2 3 5 10 20 30 40 50 60 75 100 125 150 160 170 180 185 190 195 199]
    rec_rate = 0;
    %Load test images and predict their identity.
    for i=1:40
        for j=6:10
            fname = sprintf ('att_faces/s%d/%d.pgm',i,j); %Format filename of the test image
            im = double(imread(fname));
            %Compute eigencoefficients of this image. This will be based on V as well as meanA that were computed in the training phase!
            eigcoeffs_im = V.' * (im(:) - meanA);
            
            %For current value of k, figure out the index of the closest
            %coefficient from the array eigcoeffs_training
            diff = eigcoeffs_im(1:k,:) - eigcoeffs_training(1:k,:);
            diffsqrd = diff.^2;
            diffvalue = sum(diffsqrd, 1);
            [M,I] = min(diffvalue);
            
            %Based on the index, compute the predicted identity.
            identity = id(I);
            %If the prediction matches the actual identity, increment the recognition rate rec_rate
            if identity == i
                rec_rate = rec_rate + 1;
            end
        end
    end
    rec_rate = rec_rate/Ntest;
    fprintf ('\nFor k= %d, Rec rate = %f',k,100*rec_rate);
end