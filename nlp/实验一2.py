import tkinter as tk
from tkinter import font, filedialog
import random
import math

text = """
江南 江南 江南 江南 江南 烟雨 烟雨 烟雨 烟雨 杏花 杏花 杏花 明月 明月 明月 明月 明月
清风 清风 清风 清风 清风 流云 流云 流云 荷风 荷风 荷风 竹影 竹影 竹影 松声 松声 松声
秋露 秋露 春茶 春茶 墨香 墨香 墨香 枫桥 枫桥 渔火 渔火 归舟 归舟 归舟 天涯 天涯 故里
故里 长安 长安 长安 洛阳 洛阳 东篱 东篱 采菊 采菊 南山 南山 南山 流水 流水 流水 落花
落花 落花 相思 相思 相思 相思 相思 锦瑟 锦瑟 华年 华年 庄周 蝴蝶 望帝 杜鹃 沧海 明珠
蓝田 玉烟 青苔 暮雪 残阳 孤帆 客愁 笛声 书简 墨痕 幽梦 寒潭 鹤影 松涛 竹韵 兰舟 画舫
清愁 浅唱 低吟 流年 宿命 尘缘 禅意 浮生 清欢 初见 别离 重逢 执念 淡然 留白 悠远 空灵
"""

words = text.split()
word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1

sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

max_count = max(c for _, c in sorted_words)
min_count = min(c for _, c in sorted_words)
word_sizes = {}
for word, count in sorted_words:
    size = int(12 + (80 - 12) * math.log1p(count - min_count + 1) / math.log1p(max_count - min_count + 1))
    word_sizes[word] = size

core_colors = ["#B88E8E", "#C09891", "#BFA0A0"]
scene_colors = ["#9FA8B8", "#A3ADB8", "#A9B3C0"]
aux_colors = ["#D0C8C0", "#D4CDC0", "#D2D2C8"]

core_words = {"江南", "明月", "清风", "相思", "锦瑟", "华年", "烟雨"}
scene_words = {"杏花", "荷风", "竹影", "枫桥", "渔火", "归舟", "流水", "落花", "南山"}

word_colors = {}
for word in word_sizes:
    if word in core_words:
        word_colors[word] = random.choice(core_colors)
    elif word in scene_words:
        word_colors[word] = random.choice(scene_colors)
    else:
        word_colors[word] = random.choice(aux_colors)

class DenseLiteraryWordCloud:
    def __init__(self, root):
        self.root = root
        self.root.title("密集型文艺诗词词云（正向可读）")
        self.canvas = tk.Canvas(root, width=800, height=800, bg="#F7F4F0")
        self.canvas.pack(padx=10, pady=10)
        
        self.draw_plum_blossom_contour()
        self.place_words_densely()

    def draw_plum_blossom_contour(self):
        outer_points = []
        inner_points = []
        for i in range(360):
            angle = math.radians(i)
            outer_r = 360 + 50 * math.sin(5 * angle) * math.cos(angle)
            inner_r = outer_r - 6
            outer_x = 400 + outer_r * math.cos(angle)
            outer_y = 400 + outer_r * math.sin(angle)
            inner_x = 400 + inner_r * math.cos(angle)
            inner_y = 400 + inner_r * math.sin(angle)
            outer_points.append((outer_x, outer_y))
            inner_points.append((inner_x, inner_y))
        
        for i in range(len(outer_points)-1):
            self.canvas.create_line(
                outer_points[i], outer_points[i+1], 
                fill="#7D6B5C", 
                width=2.5,
                smooth=True
            )
        for i in range(len(inner_points)-1):
            self.canvas.create_line(
                inner_points[i], inner_points[i+1], 
                fill="#A69385", 
                width=1.2,
                smooth=True
            )

    def place_words_densely(self):
        try:
            chinese_font = font.Font(family="KaiTi", size=12)
        except:
            try:
                chinese_font = font.Font(family="SimSun", size=12)
            except:
                chinese_font = font.Font(family="Microsoft YaHei", size=12)
        
        core_word_list = [(w, s) for w, s in word_sizes.items() if w in core_words]
        scene_word_list = [(w, s) for w, s in word_sizes.items() if w in scene_words]
        aux_word_list = [(w, s) for w, s in word_sizes.items() if w not in core_words and w not in scene_words]
        
        placed = []
        self.place_word_layer(core_word_list, placed, min_r=0, max_r=180, font_family=chinese_font.actual()["family"])
        self.place_word_layer(scene_word_list, placed, min_r=150, max_r=280, font_family=chinese_font.actual()["family"])
        self.place_word_layer(aux_word_list, placed, min_r=220, max_r=350, font_family=chinese_font.actual()["family"])

    def place_word_layer(self, word_list, placed, min_r, max_r, font_family):
        for word, size in word_list:
            for _ in range(200):
                angle = random.uniform(0, 2*math.pi)
                r = random.uniform(min_r, max_r)
                x = 400 + r * math.cos(angle)
                y = 400 + r * math.sin(angle)
                
                rotation = random.choice([0, 90])
                
                current_font = font.Font(family=font_family, size=size)
                word_width = current_font.measure(word)
                word_height = current_font.metrics("linespace")
                
                overlap = False
                for (px, py, pw, ph, pangle) in placed:
                    distance = math.hypot(x - px, y - py)
                    min_distance = (word_width + pw) * 0.25 + (word_height + ph) * 0.25
                    if distance < min_distance:
                        overlap = True
                        break
                if not overlap:
                    text_id = self.canvas.create_text(
                        x, y, 
                        text=word, 
                        font=current_font,
                        fill=word_colors[word], 
                        angle=rotation,
                        justify=tk.CENTER
                    )
                    placed.append((x, y, word_width, word_height, rotation))
                    break

    def save_wordcloud(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")],
            title="保存词云图"
        )
        if file_path:
            self.canvas.postscript(file=file_path + ".eps", colormode="color")
            print(f"词云图已保存为：{file_path}.eps（可用画图工具转为PNG/JPG）")

if __name__ == "__main__":
    root = tk.Tk()
    app = DenseLiteraryWordCloud(root)
    
    save_btn = tk.Button(
        root, 
        text="保存词云图", 
        command=app.save_wordcloud,
        font=font.Font(family="SimHei", size=11),
        bg="#E8E0D6",
        fg="#5D4F42",
        padx=15,
        pady=5,
        relief=tk.FLAT
    )
    save_btn.pack(pady=8)
    
    root.mainloop()