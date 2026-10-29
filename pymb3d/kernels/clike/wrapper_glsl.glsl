/* Fragment shader entry: one fragment per pixel of the rectangle, the
 * G-buffer in four RGBA32F render targets. */
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;

void main() {
    float px = floor(gl_FragCoord.x) + PARAM(P_RECT_X);
    float py = floor(gl_FragCoord.y) + PARAM(P_RECT_Y);
    Pixel g = shade_pixel(px, py);
    o0 = vec4(g.depth, g.normal.x, g.normal.y, g.normal.z);
    o1 = vec4(g.color, g.otrap, g.steps, g.ao);
    o2 = vec4(g.shadow[0], g.shadow[1], g.shadow[2], g.shadow[3]);
    o3 = vec4(g.shadow[4], g.shadow[5], g.iter, g.de);
}
