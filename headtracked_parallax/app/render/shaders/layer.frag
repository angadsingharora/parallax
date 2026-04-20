#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform float u_alpha;

void main() {
    vec4 c = texture(u_tex, v_uv);
    // Discard nearly-transparent fragments so we don't get bright fringes
    // around layer cut-outs and so depth-test doesn't write background pixels.
    if (c.a * u_alpha < 0.01) {
        discard;
    }
    frag_color = vec4(c.rgb, c.a * u_alpha);
}
