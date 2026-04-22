#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform float u_alpha;
uniform float u_neutral_mix;

void main() {
    vec4 c = texture(u_tex, v_uv);
    // Discard nearly-transparent fragments so we don't get bright fringes
    // around layer cut-outs and so depth-test doesn't write background pixels.
    if (c.a * u_alpha < 0.01) {
        discard;
    }
    // Blend toward a neutral grade: slight desaturation and mild lift.
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 neutral = mix(vec3(luma), c.rgb, 0.88) * 1.04;
    vec3 graded = mix(c.rgb, neutral, clamp(u_neutral_mix, 0.0, 1.0));
    frag_color = vec4(graded, c.a * u_alpha);
}
