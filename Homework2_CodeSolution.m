%%
% <html><h2>Problem 1</h2></html>
%%
% <html><h3>Import the data and labels</h3></html>

data = load('data3.mat').data;
x = data(:,1:2);
y = data(:,3);
%%
% <html><h3>Initial plot to see data</h3></html>

% x1 represents data with label 1
x1 = data(data(:,3)==1,1:2);
% x2 represents data with label -1
x2 = data(data(:,3)==-1,1:2);

figure;
scatter(x1(:,1),x1(:,2),'r')
hold on;
scatter(x2(:,1),x2(:,2),'b');
legend('label:1','label:-1');
xlabel('x1 i.e. first column of x');
ylabel('x2 i.e. second column of x');
title('Original Data')

%%
% <html><h3>Gradient descent set up</h3></html>

% z = x0*\theta where x0 is the feature matrix with a ones column appended
% z(i) = thetas(1)*1 + thetas(2)*x0(i,2) + thetas(3)*x0(i,3)
m = length(y);
n = length(x(1,:));
xo = [ones(m,1),x];
thetas = rand(n+1,1)*2-1;
iterations = 1000;
learning_rate = 0.1;

%%
% <html><h3>Gradient Descent</h3></html>

% errors = zeros(iterations,1);
% for i = 1:iterations
%     % get the predicted labels for the iteration
%     z = xo*thetas;
%     y_hat = ones(m,1);
%     y_hat_boolean = (z>=0);
%     y_hat(y_hat_boolean==0) = -1;
%     % get the misclassified values indices
%     misclassified_indices = (y_hat~=y);
%     %get gradient
%     gradient = -(1/m)*sum( y(misclassified_indices).*xo(misclassified_indices,:) );
%     %update thetas
%     thetas = thetas - learning_rate*transpose(gradient);
%     % get the error for the iteration
%     temp = xo*thetas;
%     errors(i) = -(1/m)*sum( y(misclassified_indices).*temp(misclassified_indices) );
% end

%%
% <html><h3>Stochastic GD</h3></html>

errors = zeros(iterations,1);
for i = 1:iterations
    % get the predicted labels for the iteration
    z = xo*thetas;
    y_hat = ones(m,1);
    y_hat_boolean = (z>=0);
    y_hat(y_hat_boolean==0) = -1;
    % get the misclassified values indices
    misclassified_indices = (y_hat~=y);
    %get gradient
    gradient = -y(misclassified_indices).*xo(misclassified_indices,:);
    %update thetas
    thetas = thetas - transpose(sum(gradient));
    % get the error for the iteration
    temp = xo*thetas;
    errors(i) = -(1/m)*sum( y(misclassified_indices).*temp(misclassified_indices) );
end

%%
% <html><h3>Plotting errors against iteration</h3></html>

figure;
plot(errors)
title('Error v. #iteration')

%%
% <html><h3>Plot decision boundary</h3></html>

x1_lin = linspace(min(x(:,1)),max(x(:,1)),1000);
x2_lin = -(thetas(1) + thetas(2)*x1_lin)/thetas(3);

figure;
scatter(x1(:,1),x1(:,2),'r')
hold on;
scatter(x2(:,1),x2(:,2),'b');
hold on;
plot(x1_lin,x2_lin,'g-');
legend('label:1','label:-1','desicion boundary')
    