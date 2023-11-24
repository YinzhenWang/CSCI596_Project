import glfw
from OpenGL.GL import *
import numpy as np

from PIL import Image, ImageDraw, ImageFont

# 创建一个字体对象，这里使用默认的字体和大小
font = ImageFont.load_default()

def create_text_image(text):
    # 创建一个新的PIL图像和绘图对象
    size = font.getsize(text)
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # 使用白色将文本绘制到图像上
    draw.text((0, 0), text, fill=(255, 255, 255), font=font)
    return image

def draw_text(text, x, y):
    # 使用Pillow创建文本图像
    image = create_text_image(text)
    # 将PIL图像转换为可以用作OpenGL纹理的数据
    text_data = image.tobytes("raw", "RGBA", 0, -1)
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 红色

    # 生成纹理
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    # 绘制纹理到屏幕上的矩形区域
    pad = 5  # 增加边距
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x - pad, y - pad)
    glTexCoord2f(0, 1); glVertex2f(x - pad, y + image.height + pad)
    glTexCoord2f(1, 1); glVertex2f(x + image.width + pad, y + image.height + pad)
    glTexCoord2f(1, 0); glVertex2f(x + image.width + pad, y - pad)
    glEnd()
    
    # 清理纹理
    glDeleteTextures(1, [texture])

def get_text_width(text):
    return font.getsize(text)[0]

def get_text_size(text):
    return font.getsize(text)

def initialize_window():
    # 初始化GLFW
    if not glfw.init():
        raise Exception("glfw can not be initialized!")

    # 创建GLFW窗口
    window = glfw.create_window(720, 720, "BERT Attention Visualization", None, None)

    # 检查窗口是否成功创建
    if not window:
        glfw.terminate()
        raise Exception("glfw window can not be created!")

    # 设置当前窗口的上下文
    glfw.make_context_current(window)
    
    # 设置视口和投影
    glViewport(0, 0, 720, 720)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 720, 0, 720, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # 设置背景颜色
    glClearColor(0.1, 0.1, 0.1, 1)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return window

def draw_line(x1, y1, x2, y2, weight):
    glLineWidth(max(weight * 10, 1))  # 线宽基于权重，最小为1
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

def render(window, tokens, attention_matrix):
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)

        # 设置文本的起始位置
        x_offset = 50  # x轴偏移量
        top_row_y = 100  # 顶行y坐标
        bottom_row_y = 620  # 底行y坐标

        token_positions = {}

        # 先绘制所有的tokens并存储位置
        for i, token in enumerate(tokens):
            text_width, text_height = get_text_size(token)
            draw_text(token, x_offset, top_row_y)  # 绘制顶行
            token_positions[i] = (x_offset + text_width / 2, top_row_y + text_height / 2)  # 存储顶行文本中心位置
            draw_text(token, x_offset, bottom_row_y)  # 绘制底行
            token_positions[i + len(tokens)] = (x_offset + text_width / 2, bottom_row_y - text_height / 2)  # 存储底行文本中心位置
            x_offset += text_width + 20  # 更新x轴偏移量
        
        # 绘制注意力矩阵
        for i, token1 in enumerate(tokens):
            for j, token2 in enumerate(tokens):
                weight = attention_matrix[i][j]
                # 设置颜色基于权重（您可能需要调整颜色以匹配您的具体权重范围）
                glColor3f(weight, weight, weight)  # 假设权重在0到1之间
                # 获取每个token的中心位置
                start_x, start_y = token_positions[i]
                end_x, end_y = token_positions[j + len(tokens)]
                draw_line(start_x, start_y, end_x, end_y, weight)
        

        glfw.swap_buffers(window)

    glfw.terminate()
