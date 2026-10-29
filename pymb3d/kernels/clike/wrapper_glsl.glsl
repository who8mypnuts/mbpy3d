/* Fragment shader entry: one fragment per pixel of the rectangle, the
 * G-buffer in four RGBA32F render targets. */
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;

void main() {
    float px = floor(gl_FragCoord.x) * PARAM(P_PIXEL_STEP) + PARAM(P_RECT_X);
    float py = floor(gl_FragCoord.y) * PARAM(P_PIXEL_STEP) + PARAM(P_RECT_Y);
    Pixel g = shade_pixel(px, py);
    o0 = vec4(g.depth, g.normal.x, g.normal.y, g.normal.z);
    o1 = vec4(g.color, g.otrap, g.steps, g.ao);
    o2 = vec4(g.shadow[0], g.shadow[1], g.shadow[2], g.shadow[3]);
    o3 = vec4(g.shadow[4], g.shadow[5], g.iter, g.rough);
    o4 = vec4(g.vol, g.r_depth, g.r_normal.x, g.r_normal.y);
    o5 = vec4(g.r_normal.z, g.r_color, g.r_otrap, g.r_ao);
    o6 = vec4(g.r_shadow[0], g.r_shadow[1], g.r_shadow[2], g.r_shadow[3]);
    o7 = vec4(g.r_shadow[4], g.r_shadow[5], g.r_steps, 0.0);
}
