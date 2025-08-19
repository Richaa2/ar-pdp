#version 300 es
precision mediump float;
out vec4 outColor;

void main() {

    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);


    if (dist > 0.5) {
        discard;
    }

    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}