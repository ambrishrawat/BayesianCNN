m = csvread('stats/statsb_18_5_1600.txt',1,0);
f = figure();
plot(m(:,1)',m(:,2)');
hold on
plot(m(:,1)',m(:,3)');
hold off
legend('cnn','bayesian cnn')
title('error rate (original label)')
xlabel('iterations')
ylabel('fraction missclassification (error rate)')
saveas(f,'stats/orig_cnn_1.jpg')


f = figure();
plot(m(:,1)',m(:,4)');
hold on
plot(m(:,1)',m(:,5)');
hold off
legend('cnn','bayesian cnn')
title('error rate (adversarial label)')
xlabel('iterations')
ylabel('fraction missclassification (error rate)')
saveas(f,'stats/adv_cnn_1.jpg')



f = figure();
scatter(m(:,6)',m(:,2)');
hold on
scatter(m(:,6)',m(:,3)');
hold off
legend('cnn','bayesian cnn')
title('error rate (original label)')
xlabel('l2-norm')
ylabel('fraction missclassification (error rate)')
saveas(f,'stats/orig_cnn_rms_1.jpg')

f = figure();
scatter(m(:,6)',m(:,4)');
hold on
scatter(m(:,6)',m(:,5)');
hold off
legend('cnn','bayesian cnn')
title('error rate (adversarial label)')
xlabel('l2-norm')
ylabel('fraction missclassification (error rate)')
saveas(f,'stats/adv_cnn_rms_1.jpg')