
mat_Z = load('./output/1.mat');

Z = mat_Z.depth_mat;
Z=double(Z);

Z = Z*1;
%Z(Z<600)=0;
figure;
set(gcf,'color','k');
sub_2 = subplot(1,1,1);
h=surf(fliplr(Z));
set(h,'FaceColor',[.7 .8 1]);
set(h,'LineStyle','none')
axis equal
grid off
box on
axis off
view(-180,90)
light('Position',[0 0 1]);lighting phong;material([.5 .6 .1]);
title('Reconstruction Output','Color', 'w');



