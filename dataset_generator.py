import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import random
import math


# Probability configuration for time index styles
P_TICKS   = 0.40   # 40% chance to draw ticks
P_NUMBERS = 0.40   # 40% chance to draw numbers
P_ROMAN   = 0.20   # 20% chance to draw Roman numerals

# Tick style probabilities
P_TICK_12 = 0.50   # 50% of ticks use 12 major marks
P_TICK_60 = 0.50   # 50% use 60 ticks
P_TICK_CIRCLES = 0.20  # 20% of ticks drawn as circles
P_TICK_LINES   = 0.80  # 80% as line markers

# Number style choices
P_NUM_4  = 0.3     # 30% of numeric clocks use only 12/3/6/9
P_NUM_12 = 0.7     # 70% use all 12

# Roman numeric choices
P_ROM_4  = 0.3
P_ROM_12 = 0.7

# New probability: clock has BOTH ticks and numbers/roman
P_COMBO = 0.40   # 40% of clocks will have both tick marks + numbers/roman

# Optional: chance of having NO indices at all
P_NO_INDICES = 0.01   # 1% chance to draw nothing (modern minimalist clocks)


# Brand labeling probability
P_BRAND = 0.35  # 35% of clocks have branding

# Brand placement probabilities
P_BRAND_ABOVE = 0.70   # 70% above center (normal wall clock)
P_BRAND_BELOW = 0.20   # 20% below center
P_BRAND_RIGHT = 0.10   # 10% to the right (watch-style)

BRAND_NAMES = [
    "TIMEX",
    "SEIKO",
    "CITIZEN",
    "CASIO",
    "OMEGA",
    "ROLEX",
    "BULOVA",
    "FOSSIL",
    "SWISS",
    "QUARTZ"
]

from PIL import ImageFont

def load_random_font(size):
    mac_fonts = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
    ]

    paths = [f for f in mac_fonts if os.path.exists(f)]
    if not paths:
        return ImageFont.load_default()

    font_path = random.choice(paths)
    return ImageFont.truetype(font_path, size)


def angle_to_sin_cos(angle_deg):
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)


# ============================================================
#          WALL BACKGROUND GENERATOR (as before)
# ============================================================

def generate_wall_background(
        width=512,
        height=512,
        gradient_strength=0.15,
        vignette_strength=0.25,
        texture_probability=0.2,
        texture_amount=0.02
    ):

    use_white = random.random() < 0.5

    if use_white:
        off_white_palette = [
            (255, 255, 255),
            (250, 250, 245),
            (245, 245, 239),
            (248, 248, 240),
            (242, 246, 255)
        ]
        base_color = np.array(random.choice(off_white_palette), dtype=np.uint8)
    else:
        base_hue = np.random.uniform(0, 1)
        base_sat = np.random.uniform(0.10, 0.35)
        base_val = np.random.uniform(0.80, 1.0)

        def hsv_to_rgb(h, s, v):
            i = int(h * 6)
            f = h * 6 - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            i = i % 6
            table = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
            return tuple(int(x * 255) for x in table[i])

        base_color = np.array(hsv_to_rgb(base_hue, base_sat, base_val), dtype=np.uint8)

    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        factor = 1 + gradient_strength * np.sin(y / height * np.pi)
        gradient[y, :, :] = base_color * factor
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)

    img_arr = gradient.astype(np.float32)

    if random.random() < texture_probability:
        noise = np.random.normal(
            0,
            255 * texture_amount,
            (height, width, 1)
        )
        noise = np.repeat(noise, 3, axis=2)
        img_arr = np.clip(img_arr + noise, 0, 255)

    img = Image.fromarray(img_arr.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    xv, yv = np.meshgrid(
        np.linspace(-1, 1, width),
        np.linspace(-1, 1, height)
    )
    distance = np.sqrt(xv ** 2 + yv ** 2)
    vignette_mask = 1 - vignette_strength * np.clip(distance, 0, 1)
    vignette_mask = vignette_mask[:, :, None]

    img_arr = np.array(img, dtype=np.float32)
    img_arr *= vignette_mask
    img = Image.fromarray(img_arr.astype(np.uint8))

    return img  # return PIL image instead of saving only


# ============================================================
#                 CLOCK FACE DRAWING FUNCTION
# ============================================================

import math
import random
from PIL import Image, ImageDraw

import math
import random
from PIL import Image, ImageDraw

def random_clock_face_color():
    mode = random.random()

    # 40% off-white variations
    if mode < 0.40:
        base = random.randint(238, 255)
        tint = random.randint(-8, 8)
        return (
            np.clip(base + tint, 230, 255),
            np.clip(base + tint, 230, 255),
            np.clip(base + tint, 230, 255)
        )

    # 30% warm / cream tones
    elif mode < 0.70:
        r = random.randint(240, 255)
        g = random.randint(232, 250)
        b = random.randint(215, 235)
        return (r, g, b)

    # 20% cool / slightly bluish tones
    elif mode < 0.90:
        base = random.randint(235, 250)
        return (base, base, random.randint(240, 255))

    # 10% very subtle pastel colors
    else:
        base = random.randint(240, 255)
        tint = random.randint(210, 255)
        channel = random.choice([0,1,2])
        rgb = [base, base, base]
        rgb[channel] = tint
        return tuple(rgb)


def draw_outer_rim_shadow(draw, cx, cy, r, strength=0.25):
    """
    Subtle shadow outside the clock rim, fading outward.
    """
    layers = 6
    for i in range(layers):
        alpha = int(255 * strength * (1 - i / layers))
        rr = r + i + 1
        draw.ellipse(
            (cx - rr, cy - rr, cx + rr, cy + rr),
            outline=(0, 0, 0, alpha),
            width=2
        )


def draw_inner_bevel(draw, cx, cy, r, strength=0.20):
    """
    Thin inner dark ring to simulate bevel depth inside rim.
    """
    # Slightly smaller than rim
    r1 = r - 2
    r2 = r - 4

    dark = (50, 50, 50)
    light = (200, 200, 200)

    # dark bottom arc
    draw.arc((cx - r1, cy - r1, cx + r1, cy + r1),
             start=70, end=250,
             fill=dark, width=3)

    # light upper arc (highlight)
    draw.arc((cx - r2, cy - r2, cx + r2, cy + r2),
             start=250, end=430,
             fill=light, width=2)


def add_clock_face(
        background_img,
        radius_ratio=0.40,
        face_fill=None,
        offcenter_probability=0.25,
        max_offset_ratio=0.08
    ):

    img = background_img.copy()
    draw = ImageDraw.Draw(img)

    # Choose clock-face color
    if face_fill is None:
        face_fill = random_clock_face_color()

    w, h = img.size
    r = int(min(w, h) * radius_ratio)

    cx, cy = w // 2, h // 2

    # -----------------------------------------------
    # RANDOM OFFCENTER
    # -----------------------------------------------
    if random.random() < offcenter_probability:
        dx = int(w * max_offset_ratio)
        dy = int(h * max_offset_ratio)
        cx += random.randint(-dx, dx)
        cy += random.randint(-dy, dy)

    # -----------------------------------------------
    # OUTER FACE
    # -----------------------------------------------
    outline_width = random.randint(4, 14)
    outline_color = random.choice([
        (0,0,0),
        (90,60,30),
        (140,110,80),
        (80,80,80),
        (150,150,150)
    ])

    bbox = (cx-r, cy-r, cx+r, cy+r)
    draw.ellipse(bbox, fill=face_fill, outline=outline_color, width=outline_width)

    # Rim shading
    draw_outer_rim_shadow(draw, cx, cy, r + outline_width//2)
    draw_inner_bevel(draw, cx, cy, r)

    # -----------------------------------------------
    # INDEX HELPERS
    # -----------------------------------------------
    def draw_tick_marks(draw, cx, cy, r):
        count = 12 if random.random() < P_TICK_12 else 60
        use_circles = random.random() < P_TICK_CIRCLES
        if use_circles:
            count = 12

        tick_len_major = r * 0.14
        tick_len_minor = r * 0.07
        
        for i in range(count):
            angle = math.radians(i/count * 360 - 90)
            vx, vy = math.cos(angle), math.sin(angle)

            if count == 60 and i % 5 != 0:
                length = tick_len_minor
                width = 3
            else:
                length = tick_len_major
                width = 4

            x1 = cx + vx * (r - length)
            y1 = cy + vy * (r - length)
            x2 = cx + vx * (r*0.98)
            y2 = cy + vy * (r*0.98)

            if use_circles:
                rr = width * 1.5
                cx2 = cx + vx * (r * 0.85)
                cy2 = cy + vy * (r * 0.85)
                draw.ellipse((cx2-rr, cy2-rr, cx2+rr, cy2+rr), fill=(0,0,0))
            else:
                draw.line((x1,y1,x2,y2), fill=(0,0,0), width=width)


    def draw_numeric_indices(draw, cx, cy, r):
        count = 4 if random.random() < P_NUM_4 else 12
        font = load_random_font(int(r*0.18))

        if count == 4:
            indices = {12:12,3:3,6:6,9:9}
        else:
            indices = {k:k for k in range(1,13)}

        for pos,val in indices.items():
            angle = math.radians((pos%12)/12 * 360 - 90)
            vx,vy = math.cos(angle),math.sin(angle)

            tx = cx + vx*(r*0.72)
            ty = cy + vy*(r*0.72)

            text = str(val)
            bbox = draw.textbbox((0,0), text, font=font)
            tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.text((tx-tw/2, ty-th/1.1), text, font=font, fill=(0,0,0))


    ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",7:"VII",8:"VIII",9:"IX",10:"X",11:"XI",12:"XII"}

    def draw_roman_indices(draw, cx, cy, r):
        count = 4 if random.random() < P_ROM_4 else 12
        font = load_random_font(int(r*0.15))

        if count == 4:
            indices = {12:"XII",3:"III",6:"VI",9:"IX"}
        else:
            indices = ROMAN

        for pos,val in indices.items():
            angle = math.radians((pos%12)/12 * 360 - 90)
            vx,vy = math.cos(angle),math.sin(angle)

            tx = cx + vx*(r*0.72)
            ty = cy + vy*(r*0.72)

            bbox = draw.textbbox((0,0), val, font=font)
            tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.text((tx-tw/2, ty-th/1.1), val, font=font, fill=(0,0,0))


    # -----------------------------------------------
    # APPLY INDEX STYLE
    # -----------------------------------------------
    coin = random.random()

    if coin < P_NO_INDICES:
        pass
    elif coin < P_NO_INDICES + P_COMBO:
        draw_tick_marks(draw, cx, cy, r)
        (draw_numeric_indices if random.random()<0.5 else draw_roman_indices)(draw,cx,cy,r)
    else:
        p = random.random()
        if p < P_TICKS:
            draw_tick_marks(draw,cx,cy,r)
        elif p < P_TICKS + P_NUMBERS:
            draw_numeric_indices(draw,cx,cy,r)
        else:
            draw_roman_indices(draw,cx,cy,r)


    # ------------------------------------------------------
    # BRAND LABEL (properly separated!)
    # ------------------------------------------------------
    def draw_brand_label(draw, img, cx, cy, r):
        brand = random.choice(BRAND_NAMES)

        # Choose placement
        p = random.random()
        if p < P_BRAND_ABOVE:
            tx = cx
            ty = cy - r*random.uniform(0.18,0.28)
        elif p < P_BRAND_ABOVE + P_BRAND_BELOW:
            tx = cx
            ty = cy + r*random.uniform(0.18,0.28)
        else:
            tx = cx + r*random.uniform(0.18,0.32)
            ty = cy

        # font
        font_size = int(r*random.uniform(0.08,0.12))
        font = load_random_font(font_size)

        bbox = draw.textbbox((0,0), brand, font=font)
        tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]

        temp = Image.new("RGBA", (tw*4, th*4), (0,0,0,0))
        tdraw = ImageDraw.Draw(temp)
        tdraw.text((temp.width//2 - tw//2, temp.height//2 - th//2),
                   brand, font=font, fill=(0,0,0,255))

        rotated = temp.rotate(random.uniform(-0.4,0.4), expand=True)

        img.paste(rotated,
                  (int(tx-rotated.width/2), int(ty-rotated.height/2)),
                  rotated)

    if random.random() < P_BRAND:
        draw_brand_label(draw, img, cx, cy, r)

    # ------------------------------------------------------
    # GENERATE TIME (correct order!)
    # ------------------------------------------------------
    hour = random.randint(0,11)
    minute = random.randint(0,59)

    minute_angle = (minute/60)*360
    hour_angle = (hour/12)*360 + (minute/60)*30

    def angle_to_vec(angle):
        a = math.radians(angle-90)
        return math.cos(a), math.sin(a)

    # ------------------------------------------------------
    # BACKWARD HAND EXTENSION
    # ------------------------------------------------------
    EXTEND_BACK_PROB = 0.25
    EXTEND_BACK_SCALE = random.uniform(0.10, 0.25)
    extend_back = (random.random() < EXTEND_BACK_PROB)

    # ------------------------------------------------------
    # HAND DRAW FUNCTIONS (using new angles)
    # ------------------------------------------------------
    def draw_rect_hand(length, width, angle_deg):
        vx,vy = angle_to_vec(angle_deg)

        x_tip = cx + vx*length
        y_tip = cy + vy*length

        if extend_back:
            back = length * EXTEND_BACK_SCALE
            x_back = cx - vx*back
            y_back = cy - vy*back
        else:
            x_back = cx
            y_back = cy

        draw.line((x_back,y_back,x_tip,y_tip), fill=(0,0,0), width=width)

    def draw_capsule_hand(length, width, angle_deg):
        vx,vy = angle_to_vec(angle_deg)

        x_tip = cx + vx*length
        y_tip = cy + vy*length

        if extend_back:
            back = length*EXTEND_BACK_SCALE
            x_back = cx - vx*back
            y_back = cy - vy*back
        else:
            x_back = cx
            y_back = cy

        px,py = -vy,vx
        half_w = width/2

        p1 = (x_back + px*half_w, y_back + py*half_w)
        p2 = (x_tip  + px*half_w, y_tip  + py*half_w)
        p3 = (x_tip  - px*half_w, y_tip  - py*half_w)
        p4 = (x_back - px*half_w, y_back - py*half_w)
        draw.polygon([p1,p2,p3,p4], fill=(0,0,0))

        draw.ellipse((x_tip-half_w, y_tip-half_w, x_tip+half_w, y_tip+half_w), fill=(0,0,0))
        draw.ellipse((x_back-half_w, y_back-half_w, x_back+half_w, y_back+half_w), fill=(0,0,0))

    def draw_pointed_hand(length,width,angle_deg):
        vx,vy = angle_to_vec(angle_deg)

        x_tip = cx + vx*length
        y_tip = cy + vy*length

        shaft = length*0.55
        x_base = cx + vx*shaft
        y_base = cy + vy*shaft

        if extend_back:
            back = length*EXTEND_BACK_SCALE
            x_back = cx - vx*back
            y_back = cy - vy*back
        else:
            x_back, y_back = cx, cy

        px,py = -vy,vx
        half_w = width*0.7

        p_tip = (x_tip,y_tip)
        p_left = (x_base + px*half_w, y_base + py*half_w)
        p_right = (x_base - px*half_w, y_base - py*half_w)
        draw.polygon([p_tip,p_left,p_right], fill=(0,0,0))

        small_w = width*0.35
        s_left = (x_back + px*small_w, y_back + py*small_w)
        s_right = (x_back - px*small_w, y_back - py*small_w)
        b_left = (x_base + px*small_w, y_base + py*small_w)
        b_right = (x_base - px*small_w, y_base - py*small_w)
        draw.polygon([s_left,b_left,b_right,s_right], fill=(0,0,0))

    # ------------------------------------------------------
    # CHOOSE HAND STYLE
    # ------------------------------------------------------
    style = random.choice(["rect","capsule","pointed"])

    # RANDOM SIZES
    length_scale = random.uniform(0.80,1.00)
    width_scale  = random.uniform(1.00,1.20)

    # minute hand
    m_len = r*0.90 * length_scale
    m_w   = int(6 * width_scale)

    # hour hand
    h_len = r*0.55 * length_scale
    h_w   = int(10 * width_scale)

    # ------------------------------------------------------
    # DRAW HANDS
    # ------------------------------------------------------
    if style=="rect":
        draw_rect_hand(m_len,m_w,minute_angle)
        draw_rect_hand(h_len,h_w,hour_angle)
    elif style=="capsule":
        draw_capsule_hand(m_len,m_w,minute_angle)
        draw_capsule_hand(h_len,h_w,hour_angle)
    else:
        draw_pointed_hand(m_len,m_w,minute_angle)
        draw_pointed_hand(h_len,h_w,hour_angle)
        
        
    # ------------------------------------------------------
    # SECOND HAND (not used for labels — just visual)
    # ------------------------------------------------------
    P_SECOND_HAND = 0.70  # 70% of clocks include a second hand

    if random.random() < P_SECOND_HAND:
        # Choose style
        second_color = random.choice([
            (0,0,0),
            (255,0,0),
            (255,255,255),
            (200,200,200)  # silver/steel
        ])

        second_width = random.randint(2, 4)
        second_length = r * random.uniform(0.90, 1.00)  # often reaches past ticks
        tail_length = r * random.uniform(0.10, 0.25)    # counterweight tail

        # Random second angle (0–360), not used for labels
        second_angle = random.uniform(0, 360)
        vx, vy = angle_to_vec(second_angle)

        # MAIN TIP
        x_tip = cx + vx * second_length
        y_tip = cy + vy * second_length

        # COUNTERWEIGHT TAIL
        x_tail = cx - vx * tail_length
        y_tail = cy - vy * tail_length

        # Draw main shaft
        draw.line(
            (x_tail, y_tail, x_tip, y_tip),
            fill=second_color,
            width=second_width
        )

        # Optional round tip
        if random.random() < 0.50:
            rr = second_width * 1.3
            draw.ellipse(
                (x_tip - rr, y_tip - rr, x_tip + rr, y_tip + rr),
                fill=second_color
            )

        # Optional counterweight circle
        if random.random() < 0.30:
            rr2 = second_width * 2
            draw.ellipse(
                (x_tail - rr2, y_tail - rr2, x_tail + rr2, y_tail + rr2),
                fill=second_color
            )

        # Center hub (covers joints)
        hub_r = second_width * 1.5
        draw.ellipse(
            (cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r),
            fill=second_color
        )

    return img, hour, minute, hour_angle, minute_angle


# ============================================================
#                      EXAMPLE USAGE
# ============================================================

# ============================================================
#                      EXAMPLE USAGE (FINAL)
# ============================================================

import csv

if __name__ == "__main__":
    DATA_ROOT = "/Users/mredd/Desktop/clock_dataset"
    IMAGES_DIR = os.path.join(DATA_ROOT, "images")
    LABELS_CSV = os.path.join(DATA_ROOT, "labels.csv")

    # Make output directories
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "sin_hour", "cos_hour",
            "sin_min", "cos_min",
            "hour", "minute"
        ])

        NUM_IMAGES = 10000  # dataset size

        for i in range(NUM_IMAGES):
            # background (does not need saving per-image now)
            bg = generate_wall_background(512, 512)

            filename = f"clock_{i:06d}.jpg"
            out_path = os.path.join(IMAGES_DIR, filename)

            img, hour, minute, hour_angle, minute_angle = add_clock_face(bg)

            # ---- compute label values ----
            sin_h, cos_h = angle_to_sin_cos(hour_angle)
            sin_m, cos_m = angle_to_sin_cos(minute_angle)

            # ---- save WebP image ----
            webp_path = out_path.replace(".jpg", ".webp")
            img.save(webp_path, quality=70, method=6)

            # ---- write CSV row ----
            writer.writerow([
                os.path.basename(webp_path),
                sin_h, cos_h,
                sin_m, cos_m,
                hour, minute
            ])
            
            if i % 100 == 0:
                print(f"Generated {i}/{NUM_IMAGES}")

    print("\nDataset complete!")
    print(f"Images saved in: {IMAGES_DIR}")
    print(f"CSV saved as: {LABELS_CSV}")