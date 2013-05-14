%% TEXTURE IMAGE VERSION


%% load a small textured square

points = [0.   0.   0.;
          0.75 0.   0.;
          0.75 0.75 0.;
          0.   0.75 0.]';
color = ones(4,3);
trilist = uint32([0 1 3;
                  1 2 3]');
texture = uint8(zeros(64, 64, 4));
texture(:,:,4) = 255;
texture(1:16, :, 1) = 255;
texture(16:32, :, 2) = 255;
texture(32:end, :, 3) = 255;
tcoords = single([0 0;
                  1 0;
                  1 1;
                  0 1]);

%%
width = 64;
height = 64;
gl = OpenGLRenderer(width, height);
[frameBuffer, coordBuffer] = gl.renderTPS(points, color, trilist + 1, tcoords, texture);
%%
imshow(frameBuffer)
%%
imshow(coordBuffer)
