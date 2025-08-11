#version 300 es
layout(location = 0) in vec3 a_Position;
uniform mat4 u_MVP;
void main() {
    // Transform the circle’s vertex into clip space
    gl_Position = u_MVP * vec4(a_Position, 1.0);
}