#version 300 es
layout(location = 0) in vec3 a_Position;
uniform mat4 u_VP;       // projection * view
void main() {
    gl_Position = u_VP * vec4(a_Position, 1.0);
}